//! Resource catalog and resolver (docs/PLAN.md, "On-demand resources"). Synthetic hardware and
//! install snapshots; no network, no runtime libraries.

use blue_onyx_prism::backend::detect::{
    ComputeCapability, CudaInfo, GpuAdapter, GpuVendor, HardwareInfo,
};
use blue_onyx_prism::backend::select::{EpStatus, OpenVinoProbe, OrtProbe, RuntimeProbe};
use blue_onyx_prism::backend::spec::Target;
use blue_onyx_prism::config::{Config, ModelConfig};
use blue_onyx_prism::model::ModelFamilyKind;
use blue_onyx_prism::resources::catalog::{
    self, CUDA_LIBS_ID, Flavor, Layout, OPENVINO_RUNTIME_ID, Platform, ResourceKind,
    SHIPPED_PLATFORMS,
};
use blue_onyx_prism::resources::resolve::{
    FETCH_FOR_CONFIG, Installed, Resolution, downloadable_options, needed,
};
use std::collections::BTreeSet;
use std::path::PathBuf;

// ---------------------------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------------------------

/// The ORT flavors `setup-onnxruntime` shipped per platform before the catalog existed.
fn expected_flavors(p: Platform) -> &'static [Flavor] {
    match (p.os, p.arch) {
        ("windows", "x86_64") => &[Flavor::Cpu, Flavor::Cuda, Flavor::DirectMl],
        ("linux", "x86_64") => &[Flavor::Cpu, Flavor::Cuda],
        ("linux", "aarch64") => &[Flavor::Cpu],
        ("macos", "aarch64") => &[Flavor::Cpu, Flavor::CoreMl],
        _ => &[],
    }
}

#[test]
fn every_shipped_platform_has_runtimes() {
    for &p in SHIPPED_PLATFORMS {
        let ov = catalog::openvino_runtime(p.os, p.arch)
            .unwrap_or_else(|| panic!("no openvino-runtime for {p}"));
        assert_eq!(ov.id, OPENVINO_RUNTIME_ID);
        assert!(ov.parts[0].url.contains(catalog::OPENVINO_VERSION), "{p}");
        let mut flavors = catalog::ort_flavors(p.os, p.arch);
        flavors.sort();
        let mut want = expected_flavors(p).to_vec();
        want.sort();
        assert_eq!(flavors, want, "ORT flavors for {p}");
        for f in flavors {
            let r = catalog::onnxruntime(p.os, p.arch, f).unwrap();
            assert_eq!(r.id, f.resource_id());
            assert_eq!(r.version, catalog::onnxruntime_version(f));
            if f == Flavor::Cuda {
                assert_eq!(r.version, catalog::ONNXRUNTIME_CUDA_VERSION);
                assert!(
                    r.parts[0]
                        .file_name
                        .contains(&format!("gpu_cuda12-{}", r.version)),
                    "{}",
                    r.parts[0].file_name
                );
            } else {
                assert_eq!(r.version, catalog::ONNXRUNTIME_VERSION);
            }
            for part in r.parts.iter().filter(|p| p.layout == Layout::OrtRelease) {
                assert!(part.url.contains(r.version), "{}", part.url);
            }
            // setup-onnxruntime reads the same table.
            let parts = blue_onyx_prism::setup_onnxruntime::packages_for(p.os, p.arch, f).unwrap();
            assert_eq!(parts.len(), r.parts.len());
            assert_eq!(parts[0].sha256, r.parts[0].sha256);
        }
        let pkg = blue_onyx_prism::setup_openvino::package_for(p.os, p.arch).unwrap();
        assert_eq!(pkg.sha256, ov.parts[0].sha256);
        let cuda = catalog::cuda_libs_for(p.os, p.arch);
        assert_eq!(
            cuda.is_some(),
            flavors_has_cuda(p),
            "nvidia-cuda-libs exactly where the CUDA flavor is: {p}"
        );
    }
    assert!(catalog::openvino_runtime("macos", "x86_64").is_none());
}

fn flavors_has_cuda(p: Platform) -> bool {
    expected_flavors(p).contains(&Flavor::Cuda)
}

#[test]
fn hashes_urls_and_sizes_are_well_formed() {
    let mut keys = BTreeSet::new();
    for r in catalog::all() {
        assert!(
            keys.insert((r.id, r.platform)),
            "duplicate {} {:?}",
            r.id,
            r.platform
        );
        assert!(!r.parts.is_empty(), "{}", r.id);
        assert!(!r.dest.is_empty() && !r.dest.starts_with('/') && !r.dest.contains(".."));
        for p in r.parts {
            assert_eq!(p.sha256.len(), 64, "{} {}", r.id, p.url);
            assert!(
                p.sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "{} sha256 must be lowercase hex: {}",
                r.id,
                p.sha256
            );
            assert!(p.url.starts_with("https://"), "{}", p.url);
            assert!(p.size > 0, "{}", p.url);
            // Hugging Face `onnx-community` exports (D-FINE, RF-DETR) are all `onnx/model.onnx`
            // and get saved under the model's name.
            let hf_export = r.kind == ResourceKind::Model && p.url.ends_with("/onnx/model.onnx");
            // Bench sets mixing several sources prefix the file name with the source
            // (`cct-<id>.jpg`) so names from different datasets cannot collide; the extension
            // may be lower-cased (`.JPG` -> `.jpg`).
            let url_name = p.url.rsplit('/').next().unwrap_or_default();
            let prefixed = r.kind == ResourceKind::BenchImages
                && p.file_name
                    .to_ascii_lowercase()
                    .ends_with(&url_name.to_ascii_lowercase());
            assert!(
                p.url.ends_with(p.file_name) || hf_export || prefixed,
                "{} vs {}",
                p.url,
                p.file_name
            );
            assert!(!p.file_name.contains('/') && !p.file_name.contains('\\'));
            assert_eq!(p.archive.is_none(), p.layout == Layout::File, "{}", p.url);
        }
    }
    // Hugging Face downloads are pinned to a commit, not `main`.
    for r in catalog::MODELS {
        for p in r.parts {
            assert!(!p.url.contains("/resolve/main/"), "{}", p.url);
            assert!(p.url.contains(r.version), "{}", p.url);
        }
    }
}

#[test]
fn url_check_covers_the_training_pins() {
    use blue_onyx_prism::resources::commands::url_check_resources;
    let checked: Vec<_> = url_check_resources().collect();
    let ids: BTreeSet<&str> = checked.iter().map(|r| r.id).collect();
    for r in catalog::TRAIN_BASE_WEIGHTS {
        assert!(ids.contains(r.id), "{} is not URL-checked", r.id);
    }
    let lila = catalog::lila_night().expect("the night training subset is pinned");
    assert!(ids.contains(lila.id), "{} is not URL-checked", lila.id);
    for r in catalog::AUTOLABEL_WEIGHTS
        .iter()
        .chain(catalog::IMPORT_DATASETS)
    {
        assert!(ids.contains(r.id), "{} is not URL-checked", r.id);
    }
    assert!(!catalog::AUTOLABEL_WEIGHTS.is_empty() && !catalog::IMPORT_DATASETS.is_empty());
    assert!(catalog::all().all(|r| ids.contains(r.id)));
    // The training, labeler and footage pins: well-formed like the rest of the catalog.
    let mut files = BTreeSet::new();
    for r in catalog::TRAIN_BASE_WEIGHTS
        .iter()
        .chain([lila])
        .chain(catalog::AUTOLABEL_WEIGHTS)
        .chain(catalog::IMPORT_DATASETS)
    {
        assert!(!r.parts.is_empty(), "{}", r.id);
        assert!(!r.dest.is_empty() && !r.dest.starts_with('/') && !r.dest.contains(".."));
        for p in r.parts {
            assert_eq!(p.sha256.len(), 64, "{} {}", r.id, p.url);
            assert!(
                p.sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "{} sha256 must be lowercase hex: {}",
                r.id,
                p.sha256
            );
            assert!(p.url.starts_with("https://"), "{}", p.url);
            assert!(p.size > 0, "{}", p.url);
            assert_eq!(p.layout, Layout::File, "{}", p.url);
            assert!(p.archive.is_none(), "{}", p.url);
            assert!(!p.file_name.contains('/') && !p.file_name.contains('\\'));
            if r.id.starts_with(catalog::AUTOLABEL_PREFIX)
                || r.id.starts_with(catalog::IMPORT_DATA_PREFIX)
            {
                assert!(p.url.ends_with(p.file_name), "{}", p.url);
            }
            assert!(
                files.insert((r.id, p.file_name)),
                "duplicate {}",
                p.file_name
            );
            // Hugging Face downloads are pinned to a commit (`hf-<40 hex>` version).
            if p.url.starts_with("https://huggingface.co/") {
                let rev = r.version.strip_prefix("hf-").unwrap_or_default();
                assert!(
                    rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()),
                    "{} version {}",
                    r.id,
                    r.version
                );
                assert!(p.url.contains(&format!("/resolve/{rev}/")), "{}", p.url);
            }
        }
    }
    // Over the large threshold: a training run asks before it downloads it.
    assert_eq!(lila.is_large(), lila.size() > catalog::LARGE_DOWNLOAD_BYTES);
}

#[test]
fn night_training_subset_is_trainable_and_disjoint_from_nir_night() {
    use blue_onyx_prism::benchmark::images::Manifest;
    use blue_onyx_prism::collect::config::DEFAULT_CLASSES;
    let m = catalog::lila_night_manifest().expect("assets/train/lila-night.json parses");
    let r = catalog::lila_night().expect("pinned");
    // Every image becomes a download part (the training reader skips missing ones).
    assert_eq!(r.parts.len(), m.images.len());
    assert_eq!(r.size(), m.size());
    assert!(m.license.starts_with("CDLA-Permissive-1.0"));
    assert!(m.attribution.contains("LILA BC"));
    let mut empty = 0usize;
    let mut cameras = BTreeSet::new();
    for i in &m.images {
        let objects = i
            .objects
            .as_ref()
            .unwrap_or_else(|| panic!("{} unannotated", i.file));
        let (w, h) = (
            i.width.expect("width") as f32,
            i.height.expect("height") as f32,
        );
        // The reader drops images with ignore regions: there must be none.
        for o in objects {
            assert!(!o.ignore && !o.crowd, "{}", i.file);
            assert!(DEFAULT_CLASSES.contains(&o.label.as_str()), "{}", o.label);
            let [x0, y0, x1, y1] = o.bbox;
            assert!(0.0 <= x0 && x0 < x1 && x1 <= w && 0.0 <= y0 && y0 < y1 && y1 <= h);
        }
        assert!(i.tags.iter().any(|t| t == "ir"), "{}", i.file);
        let cam = i.tags.iter().find_map(|t| t.strip_prefix("camera:"));
        cameras.insert(cam.unwrap_or_else(|| panic!("{} has no camera tag", i.file)));
        if objects.is_empty() {
            empty += 1;
            assert!(i.tags.iter().any(|t| t == "empty"), "{}", i.file);
        }
    }
    let share = empty as f64 / m.images.len() as f64;
    assert!((0.10..=0.20).contains(&share), "empty share {share}");
    assert!(cameras.len() >= 50, "{} cameras", cameras.len());
    // No image of the nir-night benchmark set (camera locations are checked by the builder,
    // which has the source metadata).
    let nir = Manifest::parse(include_str!("../assets/bench/nir-night.json")).unwrap();
    let nir_files: BTreeSet<&str> = nir.images.iter().map(|i| i.file.as_str()).collect();
    let nir_urls: BTreeSet<&str> = nir.images.iter().filter_map(|i| i.url.as_deref()).collect();
    for i in &m.images {
        assert!(!nir_files.contains(i.file.as_str()), "{}", i.file);
        assert!(!nir_urls.contains(i.url.as_deref().unwrap_or_default()));
    }
}

#[test]
fn large_downloads_are_only_the_cuda_libs() {
    for r in catalog::all() {
        assert_eq!(
            r.is_large(),
            r.id == CUDA_LIBS_ID,
            "{} ({} bytes)",
            r.id,
            r.size()
        );
    }
    assert_eq!(catalog::LARGE_DOWNLOAD_BYTES, 500 * 1024 * 1024);
}

#[test]
fn download_models_reads_the_catalog() {
    // `download-models` / `list-models` use the catalog's model table directly.
    let dl = blue_onyx_prism::download::catalog();
    assert_eq!(dl.len(), catalog::MODELS.len());
    for r in dl {
        let name = r.model_name().unwrap();
        assert_eq!(blue_onyx_prism::download::find(name).unwrap().id, r.id);
        assert!(!r.description.is_empty(), "{name}");
        let files: Vec<&str> = r.parts.iter().map(|p| p.file_name).collect();
        // D-FINE / RF-DETR have no class file (COCO-80 fallback).
        let detr = matches!(
            r.provides,
            catalog::Provides::Model {
                family: ModelFamilyKind::Detr | ModelFamilyKind::RfDetr,
                ..
            }
        );
        if detr {
            assert_eq!(files, [format!("{name}.onnx")]);
        } else {
            assert_eq!(files, [format!("{name}.onnx"), format!("{name}.yaml")]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Resolver
// ---------------------------------------------------------------------------------------------

fn gpu(vendor: GpuVendor, name: &str, vram_mb: u64, index: u32, discrete: bool) -> GpuAdapter {
    GpuAdapter {
        vendor,
        name: name.into(),
        vram_mb,
        index,
        discrete,
        cuda: None,
    }
}

fn intel_igpu(index: u32) -> GpuAdapter {
    gpu(
        GpuVendor::Intel,
        "Intel(R) UHD Graphics 630",
        0,
        index,
        false,
    )
}

fn rtx(index: u32) -> GpuAdapter {
    gpu(
        GpuVendor::Nvidia,
        "NVIDIA GeForce RTX 3060",
        12288,
        index,
        true,
    )
}

fn win(gpus: Vec<GpuAdapter>) -> HardwareInfo {
    HardwareInfo::new("windows", "x86_64", gpus)
}

fn mac() -> HardwareInfo {
    HardwareInfo::new(
        "macos",
        "aarch64",
        vec![gpu(GpuVendor::Apple, "Apple M2 GPU", 0, 0, false)],
    )
}

fn model(path: &str) -> ModelConfig {
    ModelConfig {
        path: PathBuf::from(path),
        ..ModelConfig::default()
    }
}

fn config(models: Vec<ModelConfig>) -> Config {
    Config {
        models,
        ..Config::default()
    }
}

fn ipcam() -> Config {
    config(vec![model("models/IPcam-general.onnx")])
}

/// Model files present (as written in the config), no runtimes.
fn files_only(cfg: &Config) -> Installed {
    let mut files = Vec::new();
    for m in &cfg.models {
        files.push(m.path.clone());
        files.push(m.path.with_extension("yaml"));
    }
    Installed {
        ort_in_build: true,
        ..Installed::default()
    }
    .with_files(&files)
}

fn ids(r: &Resolution) -> Vec<&str> {
    r.needs.iter().map(|n| n.id()).collect()
}

fn pick(r: &Resolution, model: &str) -> String {
    r.model(model)
        .and_then(|m| m.device)
        .map(|d| d.to_string())
        .unwrap_or_else(|| format!("none: {:?}", r.model(model)))
}

#[test]
fn fresh_windows_intel_igpu_needs_openvino() {
    let cfg = ipcam();
    let r = needed(&cfg, &win(vec![intel_igpu(0)]), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:gpu");
    let n = &r.needs[0];
    assert_eq!(n.blocking_models, ["IPcam-general"]);
    assert!(!n.after_restart);
    assert_eq!(n.resource.platform, Some(Platform::WINDOWS_X64));
    assert_eq!(r.ort_flavor, None);
    assert!(r.optional.is_empty());
    assert_eq!(r.downloads().len(), 1);
}

#[test]
fn fresh_windows_nvidia_with_large_downloads_needs_cuda() {
    let mut cfg = ipcam();
    cfg.allow_large_downloads = true;
    let r = needed(&cfg, &win(vec![intel_igpu(0), rtx(1)]), &files_only(&cfg));
    // Plus OpenVINO for the plan's CPU fallback (blocks nothing).
    assert_eq!(
        ids(&r),
        ["onnxruntime-cuda", CUDA_LIBS_ID, OPENVINO_RUNTIME_ID]
    );
    assert!(r.needs[2].blocking_models.is_empty());
    assert!(
        r.needs[2].reason.contains("CPU fallback"),
        "{}",
        r.needs[2].reason
    );
    assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
    assert_eq!(r.ort_flavor, Some(Flavor::Cuda));
    assert!(!r.ort_restart_required());
    assert!(r.needs.iter().all(|n| !n.after_restart));
    assert_eq!(r.needs[0].device.unwrap().to_string(), "ort:cuda:0");
}

#[test]
fn fresh_windows_nvidia_without_large_downloads_falls_to_directml() {
    let cfg = ipcam();
    // NVIDIA + Intel iGPU: the discrete NVIDIA GPU via DirectML before the iGPU, plus OpenVINO
    // for the CPU fallback.
    let r = needed(&cfg, &win(vec![intel_igpu(0), rtx(1)]), &files_only(&cfg));
    assert_eq!(ids(&r), ["onnxruntime-directml", OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "ort:directml:1");
    assert_eq!(r.ort_flavor, Some(Flavor::DirectMl));
    // The skipped CUDA option is offered as an optional (large) download.
    let opt: Vec<&str> = r.optional.iter().map(|n| n.id()).collect();
    assert_eq!(opt, ["onnxruntime-cuda", CUDA_LIBS_ID]);
    let skipped = &r.model("IPcam-general").unwrap().skipped;
    assert!(
        skipped[0].starts_with("ort:cuda:0: needs NVIDIA CUDA 12 + cuDNN 9.10 libraries"),
        "{skipped:?}"
    );
    assert!(skipped[0].contains("allow_large_downloads"));

    // NVIDIA only: DirectML too.
    let r = needed(&cfg, &win(vec![rtx(0)]), &files_only(&cfg));
    assert_eq!(ids(&r), ["onnxruntime-directml", OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "ort:directml:0");

    // Linux has no DirectML: the Intel GPU via OpenVINO.
    let linux = HardwareInfo::new("linux", "x86_64", vec![intel_igpu(0), rtx(1)]);
    let r = needed(&cfg, &linux, &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:gpu");
}

/// An NVIDIA GPU the CUDA driver reports with compute capability `major.minor`.
fn nvidia_cc(name: &str, index: u32, major: u32, minor: u32) -> GpuAdapter {
    GpuAdapter {
        cuda: Some(CudaInfo {
            ordinal: 0,
            compute_capability: ComputeCapability::new(major, minor),
        }),
        ..gpu(GpuVendor::Nvidia, name, 8060, index, true)
    }
}

fn pascal(index: u32) -> GpuAdapter {
    nvidia_cc("NVIDIA GeForce GTX 1070 Ti", index, 6, 1)
}

#[test]
fn pascal_gpu_plans_cuda_on_windows() {
    // GTX 1070 Ti + UHD 630: the catalog's 1.31.0 CUDA build has Pascal kernels, so with large
    // downloads allowed the NVIDIA GPU runs CUDA; TensorRT 10 never does (7.5+).
    let mut cfg = ipcam();
    cfg.allow_large_downloads = true;
    let hw = win(vec![pascal(0), intel_igpu(1)]);
    let inst = files_only(&cfg);
    let r = needed(&cfg, &hw, &inst);
    assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
    assert_eq!(
        ids(&r),
        ["onnxruntime-cuda", CUDA_LIBS_ID, OPENVINO_RUNTIME_ID]
    );
    assert_eq!(
        r.needs[0].resource.version,
        catalog::ONNXRUNTIME_CUDA_VERSION
    );
    let offers = downloadable_options(&cfg, &hw, &inst, true);
    let specs: Vec<String> = offers.iter().map(|d| d.device.to_string()).collect();
    assert!(specs.contains(&"ort:cuda:0".to_string()), "{specs:?}");
    assert!(specs.contains(&"ort:directml:0".to_string()), "{specs:?}");
    assert!(
        specs.iter().all(|s| !s.starts_with("ort:tensorrt")),
        "{specs:?}"
    );
    let cuda = offers
        .iter()
        .find(|d| d.device.to_string() == "ort:cuda:0")
        .unwrap();
    assert!(
        cuda.summary().starts_with(
            "will download ONNX Runtime CUDA/TensorRT (CUDA 12) 1.31.0 (393 MB), NVIDIA CUDA 12 \
             + cuDNN 9.10 libraries"
        ),
        "{}",
        cuda.summary()
    );
    assert!(
        cuda.summary().ends_with("; needs allow_large_downloads"),
        "{}",
        cuda.summary()
    );

    // Large downloads not allowed: DirectML runs the GPU, CUDA stays an optional download.
    let mut small = cfg.clone();
    small.allow_large_downloads = false;
    let r = needed(&small, &hw, &inst);
    assert_eq!(pick(&r, "IPcam-general"), "ort:directml:0");
    assert_eq!(ids(&r), ["onnxruntime-directml", OPENVINO_RUNTIME_ID]);
    let opt: Vec<&str> = r.optional.iter().map(|n| n.id()).collect();
    assert_eq!(opt, ["onnxruntime-cuda", CUDA_LIBS_ID]);
    let skipped = &r.model("IPcam-general").unwrap().skipped;
    assert!(
        skipped[0].starts_with("ort:cuda:0: needs NVIDIA CUDA 12 + cuDNN 9.10 libraries"),
        "{skipped:?}"
    );

    // An .xml-only model: ONNX Runtime cannot load it, so the iGPU via OpenVINO.
    let mut xml = cfg.clone();
    xml.models[0].path = "models/IPcam-general.xml".into();
    let r = needed(&xml, &hw, &files_only(&xml));
    assert_eq!(pick(&r, "IPcam-general"), "openvino:gpu");

    // TensorRT asked for explicitly: refused with the reason, no CUDA libraries planned.
    cfg.models[0].device = Some("ort:tensorrt".into());
    let r = needed(&cfg, &win(vec![pascal(0)]), &files_only(&cfg));
    assert!(
        r.needs
            .iter()
            .chain(&r.optional)
            .all(|n| n.id() != CUDA_LIBS_ID),
        "{:?}",
        ids(&r)
    );
    let plan = r.model("IPcam-general").unwrap();
    assert!(
        plan.skipped[0].starts_with(
            "ort:tensorrt: NVIDIA GeForce GTX 1070 Ti is compute capability 6.1; TensorRT 10 \
             needs 7.5+"
        ),
        "{plan:?}"
    );

    // Maxwell (5.2): no CUDA build has kernels; DirectML, as before.
    let mut cfg = ipcam();
    cfg.allow_large_downloads = true;
    let maxwell = win(vec![nvidia_cc("NVIDIA GeForce GTX 980", 0, 5, 2)]);
    let r = needed(&cfg, &maxwell, &files_only(&cfg));
    assert_eq!(pick(&r, "IPcam-general"), "ort:directml:0");
    assert!(
        r.needs
            .iter()
            .chain(&r.optional)
            .all(|n| n.id() != "onnxruntime-cuda" && n.id() != CUDA_LIBS_ID),
        "{:?}",
        ids(&r)
    );
    assert!(
        downloadable_options(&cfg, &maxwell, &files_only(&cfg), true)
            .iter()
            .all(|d| !d.device.to_string().starts_with("ort:cuda")),
        "no CUDA offer for a Maxwell card"
    );
}

#[test]
fn outdated_cuda_flavor_is_updated() {
    // DirectML active; onnxruntime/cuda holds the old 1.24.4 build (no Pascal kernels) and the
    // libraries are installed: the plan updates the CUDA flavor, used after a restart.
    let mut cfg = ipcam();
    cfg.allow_large_downloads = true;
    let hw = win(vec![pascal(0), intel_igpu(1)]);
    let mut inst = files_only(&cfg)
        .with_resources(&[
            "onnxruntime-cuda",
            "onnxruntime-directml",
            CUDA_LIBS_ID,
            OPENVINO_RUNTIME_ID,
        ])
        .with_version("onnxruntime-cuda", "1.24.4")
        .with_version("onnxruntime-directml", "1.24.4");
    inst.active_ort = Some(Flavor::DirectMl);
    let r = needed(&cfg, &hw, &inst);
    assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
    assert_eq!(ids(&r), ["onnxruntime-cuda"]);
    let n = &r.needs[0];
    assert!(n.after_restart);
    assert!(n.reason.ends_with("(update from 1.24.4)"), "{}", n.reason);
    assert!(inst.outdated(n.resource));
    assert_eq!(n.resource.version, "1.31.0");
    // The current version installed: nothing to download.
    let current = inst
        .clone()
        .with_version("onnxruntime-cuda", catalog::ONNXRUNTIME_CUDA_VERSION);
    let r = needed(&cfg, &hw, &current);
    assert!(r.is_satisfied(), "{:?}", ids(&r));
    assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
    // Installed cuda-libs of another cuDNN still work: never re-planned.
    let libs = current
        .clone()
        .with_version(CUDA_LIBS_ID, "cuda-12.8.1+cudnn-9.8.0.87");
    let r = needed(&cfg, &hw, &libs);
    assert!(r.is_satisfied(), "{:?}", ids(&r));

    // The old CUDA flavor loaded by this process: judged by its own build (CUDA refused with the
    // update hint), the update is planned for after a restart, DirectML is not downloaded.
    let mut loaded = inst.clone();
    loaded.active_ort = Some(Flavor::Cuda);
    loaded.probe = Some(RuntimeProbe {
        openvino: OpenVinoProbe::with_device_names(&["CPU".to_string(), "GPU".to_string()]),
        ort: OrtProbe::installed(
            "cuda",
            vec![
                EpStatus::usable(Target::Cuda),
                EpStatus::usable(Target::TensorRt),
                EpStatus::usable(Target::Cpu),
            ],
        )
        .with_version("1.24.4"),
    });
    let r = needed(&cfg, &hw, &loaded);
    assert_eq!(ids(&r), ["onnxruntime-cuda"]);
    assert!(r.needs[0].after_restart);
    assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
    // With the current build loaded, the live probe decides and nothing is planned.
    let mut live = loaded.clone();
    live.versions.insert(
        "onnxruntime-cuda".into(),
        catalog::ONNXRUNTIME_CUDA_VERSION.into(),
    );
    live.probe.as_mut().unwrap().ort.version = Some(catalog::ONNXRUNTIME_CUDA_VERSION.into());
    let r = needed(&cfg, &hw, &live);
    assert!(r.is_satisfied(), "{:?}", ids(&r));
    assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
}

#[test]
fn supported_or_unknown_compute_capability_keeps_cuda() {
    let mut cfg = ipcam();
    cfg.allow_large_downloads = true;
    for g in [nvidia_cc("NVIDIA GeForce RTX 3060", 1, 8, 6), rtx(1)] {
        let r = needed(&cfg, &win(vec![intel_igpu(0), g]), &files_only(&cfg));
        assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
        assert_eq!(
            ids(&r),
            ["onnxruntime-cuda", CUDA_LIBS_ID, OPENVINO_RUNTIME_ID]
        );
    }
}

#[test]
fn installed_system_cuda_needs_no_libs() {
    // The CUDA flavor is loaded and its EP works (CUDA installed system-wide).
    let cfg = ipcam();
    let mut inst = files_only(&cfg).with_resources(&["onnxruntime-cuda", OPENVINO_RUNTIME_ID]);
    inst.active_ort = Some(Flavor::Cuda);
    inst.probe = Some(RuntimeProbe {
        openvino: OpenVinoProbe::with_device_names(&["CPU".to_string()]),
        ort: OrtProbe::installed(
            "cuda",
            vec![
                EpStatus::usable(Target::Cuda),
                EpStatus::broken(Target::TensorRt, "libnvinfer missing"),
                EpStatus::usable(Target::Cpu),
            ],
        ),
    });
    let r = needed(&cfg, &win(vec![rtx(0)]), &inst);
    assert!(r.is_satisfied(), "{:?}", ids(&r));
    assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
    assert_eq!(r.ort_flavor, Some(Flavor::Cuda));
}

#[test]
fn fresh_macos_needs_coreml_flavor() {
    let cfg = ipcam();
    let r = needed(&cfg, &mac(), &files_only(&cfg));
    assert_eq!(ids(&r), ["onnxruntime-coreml", OPENVINO_RUNTIME_ID]);
    assert_eq!(r.needs[0].blocking_models, ["IPcam-general"]);
    assert!(
        r.needs[1].blocking_models.is_empty(),
        "the CPU fallback blocks nothing"
    );
    assert_eq!(pick(&r, "IPcam-general"), "ort:coreml");
    assert_eq!(r.ort_flavor, Some(Flavor::CoreMl));
    assert_eq!(r.needs[0].resource.dest, "onnxruntime/coreml");
}

#[test]
fn rtdetr_on_macos_uses_coreml() {
    // RT-DETRv2 runs on CoreML (batch pinned to 1); OpenVINO CPU stays the fallback.
    let cfg = config(vec![model("models/rt-detrv2-s.onnx")]);
    let r = needed(&cfg, &mac(), &files_only(&cfg));
    assert_eq!(ids(&r), ["onnxruntime-coreml", OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "rt-detrv2-s"), "ort:coreml");
    assert!(r.model("rt-detrv2-s").unwrap().skipped.is_empty());
}

#[test]
fn fresh_linux_cpu_and_arm() {
    let cfg = ipcam();
    for hw in [
        HardwareInfo::new("linux", "x86_64", vec![]),
        HardwareInfo::new("linux", "aarch64", vec![]),
    ] {
        let r = needed(&cfg, &hw, &files_only(&cfg));
        assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID], "{}", hw.arch);
        assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");
        assert_eq!(r.needs[0].resource.platform.unwrap().arch, hw.arch);
    }
}

#[test]
fn everything_installed_needs_nothing() {
    let cfg = ipcam();
    let inst = files_only(&cfg).with_resources(&[OPENVINO_RUNTIME_ID]);
    let r = needed(&cfg, &win(vec![intel_igpu(0)]), &inst);
    assert!(r.is_satisfied());
    assert_eq!(r.manual_message(), None);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:gpu");
}

#[test]
fn missing_model_files() {
    let cfg = ipcam();
    let inst = Installed {
        ort_in_build: true,
        ..Installed::default()
    }
    .with_resources(&[OPENVINO_RUNTIME_ID]);
    let r = needed(&cfg, &HardwareInfo::new("linux", "x86_64", vec![]), &inst);
    assert_eq!(ids(&r), ["model:IPcam-general"]);
    let n = &r.needs[0];
    assert_eq!(n.model_dir.as_deref(), Some(std::path::Path::new("models")));
    assert_eq!(n.blocking_models, ["IPcam-general"]);
    assert_eq!(n.resource.parts.len(), 2);
    assert!(
        n.reason.contains("models/IPcam-general.onnx"),
        "{}",
        n.reason
    );

    // Only the class file missing: the model resource is still needed.
    let inst = inst.with_files(&["models/IPcam-general.onnx"]);
    let r = needed(&cfg, &HardwareInfo::new("linux", "x86_64", vec![]), &inst);
    assert_eq!(ids(&r), ["model:IPcam-general"]);
    assert!(r.needs[0].reason.contains("IPcam-general.yaml"));

    // D-FINE / RF-DETR ship no class file: the model alone is enough, and a missing model
    // needs a single `.onnx` part.
    let detr = config(vec![
        model("models/dfine-s.onnx"),
        model("models/rfdetr-base.onnx"),
    ]);
    let with_onnx = inst
        .clone()
        .with_files(&["models/dfine-s.onnx", "models/rfdetr-base.onnx"]);
    let r = needed(
        &detr,
        &HardwareInfo::new("linux", "x86_64", vec![]),
        &with_onnx,
    );
    assert!(r.is_satisfied(), "{:?}", ids(&r));
    let r = needed(&detr, &HardwareInfo::new("linux", "x86_64", vec![]), &inst);
    assert_eq!(ids(&r), ["model:dfine-s", "model:rfdetr-base"]);
    assert!(r.needs.iter().all(|n| n.resource.parts.len() == 1));

    // A file that is not in the catalog cannot be fetched; YOLO26 says how to export it.
    let cfg = config(vec![
        model("models/yolo26s.xml"),
        model("models/custom.onnx"),
    ]);
    let r = needed(&cfg, &HardwareInfo::new("linux", "x86_64", vec![]), &inst);
    assert!(r.is_satisfied());
    let e = r.model("yolo26s").unwrap().error.clone().unwrap();
    assert!(
        e.contains("export_model.py") && e.contains("Config page"),
        "{e}"
    );
    assert!(
        r.model("custom")
            .unwrap()
            .error
            .as_deref()
            .unwrap()
            .contains("not in the download catalog")
    );
}

#[test]
fn ort_flavor_switch_is_active_after_restart() {
    // DirectML installed and active on an NVIDIA box; CUDA is better and allowed.
    let mut cfg = ipcam();
    cfg.allow_large_downloads = true;
    let mut inst = files_only(&cfg).with_resources(&[OPENVINO_RUNTIME_ID, "onnxruntime-directml"]);
    inst.active_ort = Some(Flavor::DirectMl);
    let r = needed(&cfg, &win(vec![rtx(0)]), &inst);
    assert_eq!(ids(&r), ["onnxruntime-cuda", CUDA_LIBS_ID]);
    assert!(
        r.needs[0].after_restart,
        "a second flavor loads after a restart"
    );
    assert!(!r.needs[1].after_restart);
    assert_eq!(r.ort_flavor, Some(Flavor::Cuda));
    assert_eq!(r.active_ort, Some(Flavor::DirectMl));
    assert!(r.ort_restart_required());

    // A user-managed ORT (`onnxruntime_dir`) is never replaced: DirectML on the NVIDIA GPU.
    inst.ort_pinned = true;
    let r = needed(&cfg, &win(vec![rtx(0)]), &inst);
    assert!(r.is_satisfied(), "{:?}", ids(&r));
    assert_eq!(pick(&r, "IPcam-general"), "ort:directml:0");
    assert!(!r.ort_restart_required());
}

#[test]
fn one_flavor_per_process_across_models() {
    // The default model pins DirectML; the second (auto) model then uses DirectML too instead of
    // asking for the CUDA flavor.
    let mut cfg = config(vec![
        ModelConfig {
            device: Some("ort:directml".into()),
            ..model("models/IPcam-general.onnx")
        },
        model("models/IPcam-dark.onnx"),
    ]);
    cfg.allow_large_downloads = true;
    let r = needed(&cfg, &win(vec![rtx(0)]), &files_only(&cfg));
    assert_eq!(ids(&r), ["onnxruntime-directml", OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "ort:directml");
    assert_eq!(pick(&r, "IPcam-dark"), "ort:directml:0");
    assert_eq!(r.needs[0].blocking_models, ["IPcam-general", "IPcam-dark"]);
    assert_eq!(r.ort_flavor, Some(Flavor::DirectMl));

    // With `default_model` naming the second model, it goes first and picks CUDA; the explicit
    // DirectML model then falls back to OpenVINO CPU.
    cfg.default_model = Some("IPcam-dark".into());
    let r = needed(&cfg, &win(vec![rtx(0)]), &files_only(&cfg));
    assert_eq!(
        ids(&r),
        ["onnxruntime-cuda", CUDA_LIBS_ID, OPENVINO_RUNTIME_ID]
    );
    assert_eq!(pick(&r, "IPcam-dark"), "ort:cuda:0");
    assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");
    assert_eq!(r.models[0].model, "IPcam-dark");
}

#[test]
fn large_downloads_disallowed_for_explicit_cuda() {
    let cfg = config(vec![ModelConfig {
        device: Some("ort:cuda".into()),
        ..model("models/IPcam-general.onnx")
    }]);
    let r = needed(&cfg, &win(vec![rtx(0)]), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");
    let opt: Vec<&str> = r.optional.iter().map(|n| n.id()).collect();
    assert_eq!(opt, ["onnxruntime-cuda", CUDA_LIBS_ID]);
    // TensorRT is never downloadable.
    let cfg = config(vec![ModelConfig {
        device: Some("ort:tensorrt".into()),
        ..model("models/IPcam-general.onnx")
    }]);
    let mut big = cfg.clone();
    big.allow_large_downloads = true;
    let r = needed(&big, &win(vec![rtx(0)]), &files_only(&cfg));
    assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");
    assert!(r.model("IPcam-general").unwrap().skipped[0].contains("TensorRT"));
}

#[test]
fn auto_download_off_reports_with_command() {
    let mut cfg = ipcam();
    cfg.auto_download = false;
    let inst = Installed {
        ort_in_build: true,
        ..Installed::default()
    };
    let r = needed(&cfg, &win(vec![intel_igpu(0)]), &inst);
    assert_eq!(ids(&r), ["model:IPcam-general", OPENVINO_RUNTIME_ID]);
    assert!(r.downloads().is_empty(), "nothing is downloaded");
    let msg = r.manual_message().unwrap();
    assert!(msg.contains(FETCH_FOR_CONFIG), "{msg}");
    assert!(msg.contains("OpenVINO runtime 2026.4.0 (235 MB)"), "{msg}");
    assert!(msg.contains("Model IPcam-general"), "{msg}");
    assert_eq!(
        r.needs[1].fetch_command(),
        "blue-onyx-prism fetch --resource openvino-runtime"
    );
    let json = serde_json::to_value(&r).unwrap();
    assert_eq!(json["needs"][1]["resource"], "openvino-runtime");
    assert_eq!(json["auto_download"], false);
}

#[test]
fn explicit_device_specs() {
    let linux = HardwareInfo::new("linux", "x86_64", vec![]);
    // ort:cpu on Linux: the plain CPU package.
    let cfg = config(vec![ModelConfig {
        device: Some("ort:cpu".into()),
        ..model("models/IPcam-general.onnx")
    }]);
    let r = needed(&cfg, &linux, &files_only(&cfg));
    assert_eq!(ids(&r), ["onnxruntime-cpu", OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "ort:cpu");

    // ort:cpu on macOS takes the CoreML package (the same archive), so CoreML stays possible.
    let r = needed(&cfg, &mac(), &files_only(&cfg));
    assert_eq!(ids(&r), ["onnxruntime-coreml", OPENVINO_RUNTIME_ID]);

    // openvino:gpu without an Intel GPU: OpenVINO CPU as the fallback.
    let cfg = config(vec![ModelConfig {
        device: Some("openvino:gpu".into()),
        ..model("models/IPcam-general.onnx")
    }]);
    let r = needed(&cfg, &mac(), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");

    // Global legacy "CPU" with force_cpu on Windows NVIDIA: OpenVINO CPU.
    let mut cfg = ipcam();
    cfg.force_cpu = true;
    cfg.allow_large_downloads = true;
    let r = needed(&cfg, &win(vec![rtx(0)]), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(pick(&r, "IPcam-general"), "openvino:cpu");

    // An `.xml` model has no ONNX for ORT: CoreML is skipped on macOS.
    let cfg = config(vec![ModelConfig {
        family: ModelFamilyKind::Yolo26,
        ..model("models/yolo26s.xml")
    }]);
    let r = needed(&cfg, &mac(), &files_only(&cfg));
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    // ...unless a sibling `.onnx` exists.
    let inst = files_only(&cfg).with_files(&["models/yolo26s.onnx"]);
    let r = needed(&cfg, &mac(), &inst);
    assert_eq!(ids(&r), ["onnxruntime-coreml", OPENVINO_RUNTIME_ID]);
}

#[test]
fn disabled_models_and_builds_without_ort() {
    let mut cfg = config(vec![
        model("models/IPcam-general.onnx"),
        ModelConfig {
            enabled: false,
            ..model("models/rt-detrv2-s.onnx")
        },
    ]);
    let mut inst = files_only(&cfg);
    inst.files.remove(&PathBuf::from("models/rt-detrv2-s.onnx"));
    inst.ort_in_build = false;
    // macOS without ONNX Runtime support in the binary: OpenVINO CPU, and the disabled model's
    // missing file is ignored.
    let r = needed(&cfg, &mac(), &inst);
    assert_eq!(ids(&r), [OPENVINO_RUNTIME_ID]);
    assert_eq!(r.models.len(), 1);
    // A live probe with OpenVINO loaded counts as installed (e.g. a system install).
    inst.probe = Some(RuntimeProbe::openvino_only(&["CPU".to_string()]));
    let r = needed(&cfg, &mac(), &inst);
    assert!(r.is_satisfied());
    cfg.models[0].device = Some("bogus:device".into());
    let r = needed(&cfg, &mac(), &inst);
    assert!(r.model("IPcam-general").unwrap().error.is_some());
}

#[test]
fn detect_installed_reads_the_layout() {
    use blue_onyx_prism::resources::resolve::detect_installed;
    let root = std::env::temp_dir().join(format!("bop-resources-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("openvino")).unwrap();
    std::fs::write(root.join("openvino/VERSION"), "2026.4.0\n").unwrap();
    std::fs::create_dir_all(root.join("onnxruntime/cuda")).unwrap();
    std::fs::write(root.join("onnxruntime/cuda/flavor.txt"), "cuda\n").unwrap();
    std::fs::create_dir_all(root.join("models")).unwrap();
    let model_path = root.join("models/IPcam-general.onnx");
    std::fs::write(&model_path, "x").unwrap();
    let cfg = config(vec![ModelConfig {
        path: model_path.clone(),
        ..ModelConfig::default()
    }]);
    let inst = detect_installed(&cfg, &root, None);
    assert!(inst.has(OPENVINO_RUNTIME_ID));
    assert!(inst.has("onnxruntime-cuda"));
    assert!(!inst.has(CUDA_LIBS_ID));
    assert!(inst.has_file(&model_path));
    assert!(!inst.has_file(&model_path.with_extension("yaml")));
    // The per-flavor dir is not what the loader picks today (it reads `onnxruntime/` flat),
    // unless ORT_DYLIB_PATH points elsewhere.
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        assert_eq!(inst.active_ort, None);
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn fetch_jobs_follow_the_resolver() {
    use blue_onyx_prism::resources::commands::{FetchRequest, fetch_jobs};
    let root = PathBuf::from("/srv/bop");
    let cfg = ipcam();
    let hw = win(vec![rtx(0)]);
    let inst = files_only(&cfg);
    let ids = |jobs: &[blue_onyx_prism::resources::Job]| -> Vec<&str> {
        jobs.iter().map(|j| j.resource.id).collect()
    };
    // Default = --for-config: DirectML for the NVIDIA GPU, OpenVINO for the CPU fallback, and a
    // note about the skipped CUDA libraries.
    let (jobs, notes) = fetch_jobs(&cfg, &hw, &inst, &root, &FetchRequest::default()).unwrap();
    assert_eq!(ids(&jobs), ["onnxruntime-directml", OPENVINO_RUNTIME_ID]);
    assert_eq!(jobs[1].target, root.join("openvino"));
    assert!(
        notes
            .iter()
            .any(|n| n.contains(CUDA_LIBS_ID) && n.contains("--allow-large"))
    );
    // --allow-large: the CUDA flavor (marked active, nothing active yet) and the libraries.
    let req = FetchRequest {
        allow_large: true,
        ..FetchRequest::default()
    };
    let (jobs, _) = fetch_jobs(&cfg, &hw, &inst, &root, &req).unwrap();
    assert_eq!(
        ids(&jobs),
        ["onnxruntime-cuda", CUDA_LIBS_ID, OPENVINO_RUNTIME_ID]
    );
    assert!(jobs[0].activate_ort);
    assert_eq!(jobs[0].target, root.join("onnxruntime/cuda"));
    assert_eq!(jobs[1].target, root.join("onnxruntime/cuda-libs"));
    // Named resources; large ones need --allow-large; unknown ids list what exists.
    let req = FetchRequest {
        resources: vec!["model:ipcam-dark".into(), "onnxruntime-directml".into()],
        ..FetchRequest::default()
    };
    let (jobs, _) = fetch_jobs(&cfg, &hw, &inst, &root, &req).unwrap();
    assert_eq!(ids(&jobs), ["model:IPcam-dark", "onnxruntime-directml"]);
    assert!(jobs[0].target.ends_with("models"));
    let req = FetchRequest {
        resources: vec![CUDA_LIBS_ID.into()],
        ..FetchRequest::default()
    };
    // Naming the large resource is the opt-in.
    let (jobs, notes) = fetch_jobs(&cfg, &hw, &inst, &root, &req).unwrap();
    assert_eq!(ids(&jobs), [CUDA_LIBS_ID]);
    assert!(notes.iter().any(|n| n.contains("NVIDIA")), "{notes:?}");
    let req = FetchRequest {
        resources: vec!["onnxruntime-coreml".into()],
        ..FetchRequest::default()
    };
    assert!(
        format!(
            "{:#}",
            fetch_jobs(&cfg, &hw, &inst, &root, &req).unwrap_err()
        )
        .contains("available: openvino-runtime")
    );
    // Everything for the platform, without the large libraries.
    let req = FetchRequest {
        all_for_platform: true,
        ..FetchRequest::default()
    };
    let (jobs, notes) = fetch_jobs(&cfg, &hw, &inst, &root, &req).unwrap();
    assert_eq!(jobs.len(), 1 + 3 + catalog::MODELS.len());
    assert!(!ids(&jobs).contains(&CUDA_LIBS_ID));
    assert_eq!(notes.len(), 1);
}

#[test]
fn detect_installed_reads_manifests() {
    use blue_onyx_prism::resources::manager::{MANIFEST_FILE, Manifest};
    use blue_onyx_prism::resources::resolve::detect_installed;
    let root = std::env::temp_dir().join(format!("bop-manifest-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let write = |dir: &str, id: &str| {
        std::fs::create_dir_all(root.join(dir)).unwrap();
        let m = Manifest {
            id: id.into(),
            version: "x".into(),
            sha256: vec![],
            files: vec![],
        };
        std::fs::write(
            root.join(dir).join(MANIFEST_FILE),
            serde_json::to_vec(&m).unwrap(),
        )
        .unwrap();
    };
    write("openvino", OPENVINO_RUNTIME_ID);
    write("onnxruntime/directml", "onnxruntime-directml");
    write("onnxruntime/cuda-libs", CUDA_LIBS_ID);
    // A manifest for something else does not count.
    write("onnxruntime/cpu", "onnxruntime-cuda");
    // A VERSION marker without a manifest (an older install) gives the version too.
    std::fs::create_dir_all(root.join("onnxruntime/coreml")).unwrap();
    std::fs::write(root.join("onnxruntime/coreml/flavor.txt"), "coreml\n").unwrap();
    std::fs::write(root.join("onnxruntime/coreml/VERSION"), "1.20.0\n").unwrap();
    let inst = detect_installed(&Config::default(), &root, None);
    assert!(inst.has(OPENVINO_RUNTIME_ID));
    assert!(inst.has("onnxruntime-directml"));
    assert!(inst.has(CUDA_LIBS_ID));
    assert!(!inst.has("onnxruntime-cpu"));
    assert_eq!(inst.installed_version("onnxruntime-directml"), Some("x"));
    assert_eq!(inst.installed_version(CUDA_LIBS_ID), Some("x"));
    assert_eq!(inst.installed_version("onnxruntime-coreml"), Some("1.20.0"));
    assert_eq!(inst.installed_version("onnxruntime-cpu"), None);
    let dml = catalog::onnxruntime("windows", "x86_64", Flavor::DirectMl).unwrap();
    assert!(inst.outdated(dml));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn cuda_libraries_system_install_vs_download() {
    let hw = win(vec![rtx(0)]);
    for allow_large in [false, true] {
        let mut cfg = ipcam();
        cfg.allow_large_downloads = allow_large;
        // A CUDA toolkit on the library path: no 1.8 GB download, large downloads or not.
        let mut inst = files_only(&cfg);
        inst.system_cuda = true;
        let r = needed(&cfg, &hw, &inst);
        assert_eq!(
            pick(&r, "IPcam-general"),
            "ort:cuda:0",
            "allow_large={allow_large}"
        );
        assert!(!ids(&r).contains(&CUDA_LIBS_ID), "{:?}", ids(&r));
        assert_eq!(ids(&r)[0], "onnxruntime-cuda");
        assert!(r.optional.is_empty());

        // No system CUDA: the libraries are needed (allowed) or the plan moves on (not allowed).
        let inst = files_only(&cfg);
        let r = needed(&cfg, &hw, &inst);
        if allow_large {
            assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
            assert!(ids(&r).contains(&CUDA_LIBS_ID));
        } else {
            assert_eq!(pick(&r, "IPcam-general"), "ort:directml:0");
            assert!(!ids(&r).contains(&CUDA_LIBS_ID));
            assert!(r.optional.iter().any(|n| n.id() == CUDA_LIBS_ID));
        }
    }
    // Installed nvidia-cuda-libs count like a system install.
    let cfg = ipcam();
    let inst = files_only(&cfg).with_resources(&[CUDA_LIBS_ID]);
    let r = needed(&cfg, &hw, &inst);
    assert_eq!(pick(&r, "IPcam-general"), "ort:cuda:0");
    assert_eq!(ids(&r), ["onnxruntime-cuda", OPENVINO_RUNTIME_ID]);
    // Linux too; never on macOS / Linux arm64 (no CUDA flavor there).
    let linux = HardwareInfo::new("linux", "x86_64", vec![rtx(0)]);
    let mut inst = files_only(&cfg);
    inst.system_cuda = true;
    assert_eq!(
        pick(&needed(&cfg, &linux, &inst), "IPcam-general"),
        "ort:cuda:0"
    );
    let arm = HardwareInfo::new("linux", "aarch64", vec![rtx(0)]);
    assert_eq!(
        pick(&needed(&cfg, &arm, &inst), "IPcam-general"),
        "openvino:cpu"
    );
}

// ---------------------------------------------------------------------------------------------
// Model export (catalog pins, needs-export, fetch routing)
// ---------------------------------------------------------------------------------------------

#[test]
fn yolo26_exports_and_uv_are_pinned() {
    // YOLO26: five sizes, unchanged. Every Ultralytics model: the official release assets,
    // AGPL-3.0, family yolo26 (end-to-end v10 / 26) or yolo8.
    let sizes: Vec<&str> = catalog::EXPORT_MODELS
        .iter()
        .map(|r| r.model_name().unwrap())
        .filter(|n| n.starts_with("yolo26"))
        .collect();
    assert_eq!(
        sizes,
        ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]
    );
    for r in catalog::EXPORT_MODELS {
        let name = r.model_name().unwrap();
        assert_eq!(r.id, format!("model:{name}"));
        assert_eq!(r.kind, ResourceKind::ExportModel);
        assert_eq!(
            catalog::export_files(r),
            Some((format!("{name}.onnx"), format!("{name}.yaml")))
        );
        assert!(catalog::model(name).is_none());
        assert!(blue_onyx_prism::download::find(name).is_none());
        if catalog::export_recipe(r) != Some(catalog::ExportRecipe::Ultralytics) {
            continue;
        }
        assert_eq!(r.license, "AGPL-3.0");
        assert!(r.platform.is_none());
        assert_eq!(r.parts.len(), 1);
        let p = &r.parts[0];
        // MegaDetector v6 (Microsoft's Zenodo record, its own file names) or the Ultralytics
        // release assets.
        let mdv6 = name.starts_with("mdv6-");
        if mdv6 {
            assert_eq!(r.version, "ultralytics-8.4.163+zenodo-15398270");
            assert!(p.file_name.starts_with("MDV6-") && p.file_name.ends_with(".pt"));
            assert_eq!(
                p.url,
                format!("https://zenodo.org/records/15398270/files/{}", p.file_name)
            );
        } else {
            assert_eq!(r.version, catalog::ULTRALYTICS_EXPORT_VERSION);
            assert_eq!(
                p.url,
                format!("https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}.pt")
            );
            assert_eq!(p.file_name, format!("{name}.pt"));
        }
        assert_eq!(p.layout, Layout::File);
        assert!(
            p.size > 4_000_000 && p.size < 150_000_000,
            "{name} {}",
            p.size
        );
        assert!(!r.is_large(), "the weights alone are small");
        let e2e = ["yolov10", "yolo26", "mdv6-yolov10"]
            .iter()
            .any(|p| name.starts_with(p));
        assert!(
            matches!(
                r.provides,
                catalog::Provides::Model {
                    family: ModelFamilyKind::Yolo26,
                    ..
                }
            ) == e2e
        );
        assert_eq!(
            catalog::export_files(r),
            Some((format!("{name}.onnx"), format!("{name}.yaml")))
        );
        // Not a download: `download-models` / the resolver never see it.
        assert!(catalog::model(name).is_none());
        assert!(blue_onyx_prism::download::find(name).is_none());
        // `fetch --resource model:<name>` finds it on every platform.
        for p in SHIPPED_PLATFORMS {
            assert_eq!(
                catalog::find(&format!("MODEL:{name}"), p.os, p.arch).map(|r| r.id),
                Some(r.id)
            );
        }
        assert_eq!(
            catalog::export_model(&format!("models/{name}.onnx"))
                .unwrap()
                .id,
            r.id
        );
    }
    assert!(catalog::export_model("yolo26q").is_none());
    assert!(catalog::export_model("IPcam-general").is_none());

    // uv: one archive per platform, binary-only extraction, hashes from the release.
    let triples = [
        (Platform::MACOS_ARM64, "uv-aarch64-apple-darwin.tar.gz"),
        (
            Platform::new("macos", "x86_64"),
            "uv-x86_64-apple-darwin.tar.gz",
        ),
        (Platform::LINUX_X64, "uv-x86_64-unknown-linux-gnu.tar.gz"),
        (Platform::LINUX_ARM64, "uv-aarch64-unknown-linux-gnu.tar.gz"),
        (Platform::WINDOWS_X64, "uv-x86_64-pc-windows-msvc.zip"),
    ];
    assert_eq!(catalog::TOOLS.len(), triples.len());
    for (p, file) in triples {
        let r = catalog::uv_for(p.os, p.arch).unwrap_or_else(|| panic!("no uv for {p}"));
        assert_eq!(r.id, catalog::UV_ID);
        assert_eq!(r.kind, ResourceKind::Tool);
        assert_eq!(r.version, catalog::UV_VERSION);
        assert_eq!(r.dest, catalog::UV_DEST);
        assert_eq!(r.parts.len(), 1);
        let part = &r.parts[0];
        assert_eq!(part.file_name, file);
        assert_eq!(
            part.url,
            format!("https://github.com/astral-sh/uv/releases/download/0.12.19/{file}")
        );
        assert_eq!(part.layout, Layout::Uv);
        let zip = file.ends_with(".zip");
        assert_eq!(
            part.archive,
            Some(if zip {
                catalog::ArchiveKind::Zip
            } else {
                catalog::ArchiveKind::TarGz
            })
        );
        assert!(part.size > 10_000_000 && part.size < 40_000_000);
        assert_eq!(
            catalog::find("tool:uv", p.os, p.arch).unwrap().id,
            "tool:uv"
        );
    }
    assert!(catalog::uv_for("linux", "riscv64").is_none());
    // Export support where the toolchain is estimated (every shipped platform).
    for p in SHIPPED_PLATFORMS {
        let e = catalog::toolchain_estimate(p.os, p.arch).unwrap();
        assert!(
            catalog::is_large(e.disk),
            "the toolchain needs a confirmation"
        );
    }
    assert!(catalog::toolchain_estimate("macos", "x86_64").is_none());
}

#[test]
fn rfdetr_large_is_an_export_now() {
    let inst = Installed {
        ort_in_build: true,
        ..Installed::default()
    }
    .with_resources(&[OPENVINO_RUNTIME_ID]);
    let hw = HardwareInfo::new("linux", "x86_64", vec![]);
    // A config that still names the old download: needs export, nothing is downloaded.
    let cfg = config(vec![
        model("models/rfdetr-large.onnx"),
        model("models/deimv2-n.onnx"),
    ]);
    let r = needed(&cfg, &hw, &inst);
    assert!(r.is_satisfied(), "{:?}", ids(&r));
    for (m, id) in [
        ("rfdetr-large", "model:rfdetr-large"),
        ("deimv2-n", "model:deimv2-n"),
    ] {
        let p = r.model(m).unwrap();
        assert_eq!(p.needs_export, Some(id));
        assert!(p.error.as_deref().unwrap().starts_with("needs export:"));
    }
    assert!(blue_onyx_prism::download::find("rfdetr-large").is_none());

    // The old onnx-community download on disk: its manifest is another version, so it does not
    // count as installed and the plan says it is outdated.
    let root = std::env::temp_dir().join(format!("bop-rfdetr-old-{}", std::process::id()));
    let models = root.join("models");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::write(models.join("rfdetr-large.onnx"), b"old v1 model").unwrap();
    std::fs::write(
        models.join(".rfdetr-large.installed.json"),
        r#"{"id":"model:rfdetr-large","version":"4988fbacee4aa815e43c2e0666377ab26bddf1e8","sha256":[],"files":["rfdetr-large.onnx"]}"#,
    )
    .unwrap();
    let cfg = Config {
        download_dir: Some(root.clone()),
        ..config(vec![model("models/rfdetr-large.onnx")])
    };
    let inst = blue_onyx_prism::resources::resolve::detect_installed(&cfg, &root, None);
    assert!(!inst.has_file(std::path::Path::new("models/rfdetr-large.onnx")));
    let r = needed(&cfg, &hw, &inst);
    let p = r.model("rfdetr-large").unwrap();
    assert_eq!(p.needs_export, Some("model:rfdetr-large"));
    assert!(
        p.error.as_deref().unwrap().contains("older version"),
        "{:?}",
        p.error
    );
    let res = catalog::export_model("rfdetr-large").unwrap();
    assert!(!blue_onyx_prism::resources::export::is_exported(
        res, &models
    ));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn missing_yolo26_needs_export_and_is_never_downloaded() {
    let inst = Installed {
        ort_in_build: true,
        ..Installed::default()
    }
    .with_resources(&[OPENVINO_RUNTIME_ID]);
    let hw = HardwareInfo::new("linux", "x86_64", vec![]);
    for auto_download in [true, false] {
        let cfg = Config {
            auto_download,
            ..config(vec![
                model("models/yolo26s.onnx"),
                model("models/yolo26n.xml"),
            ])
        };
        let r = needed(&cfg, &hw, &inst);
        assert!(r.is_satisfied(), "nothing to download: {:?}", ids(&r));
        assert!(r.manual_message().is_none());
        let s = r.model("yolo26s").unwrap();
        assert_eq!(s.needs_export, Some("model:yolo26s"));
        let e = s.error.as_deref().unwrap();
        assert!(e.starts_with("needs export: models/yolo26s.onnx"), "{e}");
        assert!(e.contains("fetch --resource model:yolo26s"), "{e}");
        // The export writes .onnx only: an .xml path is told to point at it.
        let n = r.model("yolo26n").unwrap();
        assert_eq!(n.needs_export, None);
        assert!(n.error.as_deref().unwrap().contains("yolo26n.onnx"));
    }
    // Exported: plans normally.
    let inst = inst.with_files(&["models/yolo26s.onnx", "models/yolo26s.yaml"]);
    let cfg = config(vec![model("models/yolo26s.onnx")]);
    let r = needed(&cfg, &hw, &inst);
    let s = r.model("yolo26s").unwrap();
    assert_eq!(s.needs_export, None);
    assert!(s.error.is_none() && s.device.is_some(), "{s:?}");
}

#[test]
fn fetch_routes_yolo26_to_the_exporter() {
    use blue_onyx_prism::resources::commands::{FetchRequest, fetch_jobs, requested_exports};
    let root = PathBuf::from("/srv/bop");
    let cfg = ipcam();
    let hw = win(vec![]);
    let inst = files_only(&cfg).with_resources(&[OPENVINO_RUNTIME_ID]);
    let req = FetchRequest {
        resources: vec![
            "model:yolo26s".into(),
            "MODEL:YOLO26S".into(),
            "tool:uv".into(),
        ],
        ..FetchRequest::default()
    };
    let (jobs, _) = fetch_jobs(&cfg, &hw, &inst, &root, &req).unwrap();
    let ids: Vec<&str> = jobs.iter().map(|j| j.resource.id).collect();
    assert_eq!(ids, ["tool:uv"], "the export is not a download job");
    assert_eq!(jobs[0].target, root.join("tools/uv"));
    let ex = requested_exports(&req, "windows", "x86_64");
    assert_eq!(
        ex.iter().map(|r| r.id).collect::<Vec<_>>(),
        ["model:yolo26s"]
    );
    // --all-for-platform never exports.
    let req = FetchRequest {
        all_for_platform: true,
        allow_large: true,
        ..FetchRequest::default()
    };
    let (jobs, _) = fetch_jobs(&cfg, &hw, &inst, &root, &req).unwrap();
    assert!(jobs.iter().all(|j| !matches!(
        j.resource.kind,
        ResourceKind::ExportModel | ResourceKind::Tool
    )));
    assert!(requested_exports(&req, "windows", "x86_64").is_empty());
}
