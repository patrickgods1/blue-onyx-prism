//! Model pins: the Hugging Face models downloaded as files ([`MODELS`], pinned repo commits) and
//! the models exported locally on request ([`EXPORT_MODELS`], [`ExportRecipe`]), with their
//! input sizes, f32 needs and toolchain estimates. Pure data plus small lookups; hashes are
//! reviewed pins.

use super::*;

// ---------------------------------------------------------------------------------------------
// Models (Hugging Face, pinned commits)
// ---------------------------------------------------------------------------------------------

/// `xnorpx/rt-detr2-onnx` commit the RT-DETRv2 hashes belong to.
pub(crate) const RTDETR_REVISION: &str = "c7d9ed2f629cc8279ea0f6b84e1a86f7b82f0795";
/// `xnorpx/blue-onyx-yolo5` commit the YOLOv5 hashes belong to.
pub(crate) const YOLO5_REVISION: &str = "c0c3ff5178f5692236e4c1552d6a040135204f98";

macro_rules! hf_file {
    ($repo_rev:literal, $name:literal, $ext:literal, $sha:expr, $size:expr) => {
        Part {
            url: concat!("https://huggingface.co/", $repo_rev, "/", $name, $ext),
            sha256: $sha,
            size: $size,
            file_name: concat!($name, $ext),
            archive: None,
            layout: Layout::File,
        }
    };
}

/// Every RT-DETRv2 model uses the same COCO class list.
const RTDETR_YAML_SHA: &str = "ed029ee87019fa769269fcff15eda24a15c17dcea42b70f4dd0e3ab744503e7d";
const RTDETR_YAML_SIZE: u64 = 1193;

macro_rules! rtdetr {
    ($name:literal, $desc:literal, $onnx_sha:literal, $onnx_size:literal) => {
        Resource {
            id: concat!("model:", $name),
            kind: ResourceKind::Model,
            version: RTDETR_REVISION,
            platform: None,
            parts: &[
                hf_file!(
                    "xnorpx/rt-detr2-onnx/resolve/c7d9ed2f629cc8279ea0f6b84e1a86f7b82f0795",
                    $name,
                    ".onnx",
                    $onnx_sha,
                    $onnx_size
                ),
                hf_file!(
                    "xnorpx/rt-detr2-onnx/resolve/c7d9ed2f629cc8279ea0f6b84e1a86f7b82f0795",
                    $name,
                    ".yaml",
                    RTDETR_YAML_SHA,
                    RTDETR_YAML_SIZE
                ),
            ],
            dest: "models",
            provides: Provides::Model {
                name: $name,
                family: ModelFamilyKind::RtDetr,
            },
            title: concat!("Model ", $name),
            description: $desc,
            license: "Apache-2.0",
        }
    };
}

/// A Hugging Face `onnx-community/*-ONNX` export (D-FINE, RF-DETR): the fp32 `onnx/model.onnx`
/// of `$repo` at commit `$rev`, saved as `models/<name>.onnx`. No class file: both use the COCO-80
/// fallback (`family::detr` maps RF-DETR's COCO category ids).
macro_rules! hf_detr {
    ($family:expr, $name:literal, $repo:literal, $rev:literal, $title:literal, $desc:literal, $sha:literal, $size:literal) => {
        Resource {
            id: concat!("model:", $name),
            kind: ResourceKind::Model,
            version: $rev,
            platform: None,
            parts: &[Part {
                url: concat!(
                    "https://huggingface.co/",
                    $repo,
                    "/resolve/",
                    $rev,
                    "/onnx/model.onnx"
                ),
                sha256: $sha,
                size: $size,
                file_name: concat!($name, ".onnx"),
                archive: None,
                layout: Layout::File,
            }],
            dest: "models",
            provides: Provides::Model {
                name: $name,
                family: $family,
            },
            title: $title,
            description: $desc,
            license: "Apache-2.0",
        }
    };
}

/// D-FINE (`onnx-community/dfine_*-ONNX`, 640x640 input), family `detr`. D-FINE nano
/// (`dfine_n_coco-ONNX`) is left out: its export finds almost nothing on real photos (best score
/// 0.65 for an obvious bus, no people) on both OpenVINO and ONNX Runtime.
macro_rules! dfine {
    ($($args:tt)*) => {
        hf_detr!(ModelFamilyKind::Detr, $($args)*)
    };
}

/// RF-DETR (`onnx-community/rfdetr_*-ONNX`), family `rfdetr` (ImageNet normalization). Input
/// sizes are in [`MODEL_INPUT_SIZES`].
macro_rules! rfdetr {
    ($($args:tt)*) => {
        hf_detr!(ModelFamilyKind::RfDetr, $($args)*)
    };
}

/// Square input size of catalog models whose ONNX input is dynamic (from each Hugging Face
/// repo's `preprocessor_config.json`), and of exported models that do not use 640 (the export's
/// `--imgsz`; DEIMv2 sizes from each `config.json` `eval_spatial_size`). Models not listed use 640.
pub(crate) const MODEL_INPUT_SIZES: &[(&str, u32)] = &[
    ("dfine-s", 640),
    ("dfine-m", 640),
    ("dfine-l", 640),
    ("dfine-x", 640),
    ("rfdetr-nano", 384),
    ("rfdetr-small", 512),
    ("rfdetr-base", 560),
    ("rfdetr-medium", 576),
    ("rfdetr-large", 704),
    ("deimv2-atto", 320),
    ("deimv2-femto", 416),
];

/// Models whose activations overflow half precision: OpenVINO's f16 inference (the GPU default,
/// and the CPU default on ARM) gives NaN (DEIMv2 L/X, DINOv3 ViT-S backbones) or garbage (RF-DETR
/// L) where f32 matches PyTorch. The OpenVINO backend compiles them with
/// `INFERENCE_PRECISION_HINT=f32` (a model's explicit `gpu_precision` still wins on GPU).
pub(crate) const F32_MODELS: &[&str] = &["rfdetr-large", "deimv2-l", "deimv2-x"];

/// The model at file name / name `name` (`deimv2-l.onnx`) needs f32 inference ([`F32_MODELS`]).
pub(crate) fn needs_f32(name: &str) -> bool {
    let n = name.trim();
    let stem = std::path::Path::new(n)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(n);
    F32_MODELS.iter().any(|m| m.eq_ignore_ascii_case(stem))
}

/// Input size for a catalog model (downloaded or exported), by name or file name (`rfdetr-base`,
/// `rfdetr-base.onnx`, `deimv2-atto.onnx`).
pub(crate) fn model_input_size(name: &str) -> Option<u32> {
    let name = model(name).or_else(|| export_model(name))?.model_name()?;
    MODEL_INPUT_SIZES
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|&(_, s)| s)
}

macro_rules! yolo5 {
    ($name:literal, $desc:literal, $onnx_sha:literal, $onnx_size:literal, $yaml_sha:literal, $yaml_size:literal) => {
        Resource {
            id: concat!("model:", $name),
            kind: ResourceKind::Model,
            version: YOLO5_REVISION,
            platform: None,
            parts: &[
                hf_file!(
                    "xnorpx/blue-onyx-yolo5/resolve/c0c3ff5178f5692236e4c1552d6a040135204f98",
                    $name,
                    ".onnx",
                    $onnx_sha,
                    $onnx_size
                ),
                hf_file!(
                    "xnorpx/blue-onyx-yolo5/resolve/c0c3ff5178f5692236e4c1552d6a040135204f98",
                    $name,
                    ".yaml",
                    $yaml_sha,
                    $yaml_size
                ),
            ],
            dest: "models",
            provides: Provides::Model {
                name: $name,
                family: ModelFamilyKind::Yolo5,
            },
            title: concat!("Model ", $name),
            description: $desc,
            license: "AGPL-3.0",
        }
    };
}

/// Downloadable models (the `download.rs` catalog). Models published only as PyTorch weights
/// (Ultralytics YOLO, RF-DETR L, DEIMv2) are exported locally on request ([`EXPORT_MODELS`]).
pub static MODELS: &[Resource] = &[
    rtdetr!(
        "rt-detrv2-s",
        "RT-DETRv2 small, general COCO",
        "c849b46a43925d0c24d88c7cab096fc8158ade265b7be1ae433e3effe5758a6a",
        80530290
    ),
    rtdetr!(
        "rt-detrv2-ms",
        "RT-DETRv2 medium-small, general COCO",
        "c3315cec9f1ce83956ae1600b6224ead49bf1e5f347fb6a9a2da9740beb50a64",
        125541681
    ),
    rtdetr!(
        "rt-detrv2-m",
        "RT-DETRv2 medium, general COCO",
        "a3a7bc94da99678f788f35aeb6078e1045f4f038b266316e769c09fb51abc62d",
        132661010
    ),
    rtdetr!(
        "rt-detrv2-l",
        "RT-DETRv2 large, general COCO",
        "4c580c4f579a0372c6bea171b0ff3ccea707146a34dc5cea1b817e3c0f4fc0ba",
        169218591
    ),
    rtdetr!(
        "rt-detrv2-x",
        "RT-DETRv2 extra large, general COCO",
        "810a18839401187bd94004bd722d602158302e3f35400840e7cbcfacb71d61c2",
        300394655
    ),
    dfine!(
        "dfine-s",
        "onnx-community/dfine_s_obj2coco-ONNX",
        "f69c4ca98cba7ca58aa15b3d4600867808fecf1b",
        "D-FINE small",
        "D-FINE small (Objects365+COCO), COCO 80 classes; fast, good default for CPU/iGPU",
        "b9e2e76610053aeeac3b2f1f685d8f9a1182a93a338f624b6c8cb7fb390cb532",
        41535197
    ),
    dfine!(
        "dfine-m",
        "onnx-community/dfine_m_obj2coco-ONNX",
        "a2e1133c83887da7d5a000a025594b9d69fef5c2",
        "D-FINE medium",
        "D-FINE medium (Objects365+COCO), COCO 80 classes; balanced accuracy and speed",
        "347f2faba93248c2e7500c9e604317fb391706c58a04802dd908573376dc1323",
        78624257
    ),
    dfine!(
        "dfine-l",
        "onnx-community/dfine_l_obj2coco_e25-ONNX",
        "60e6123fe13da26a6e1dafc9d608ddeb5c393dcc",
        "D-FINE large",
        "D-FINE large (Objects365+COCO), COCO 80 classes; accurate, best on a GPU",
        "76e521ed66874590456f9eeb89c2adf33f0599778cd33b1b614f3090f95845b9",
        125348332
    ),
    dfine!(
        "dfine-x",
        "onnx-community/dfine_x_obj2coco-ONNX",
        "4d3a85a6c29f55a5346fabaf9728f5f3751f947a",
        "D-FINE extra large",
        "D-FINE extra large (Objects365+COCO), COCO 80 classes; most accurate, slow without a GPU",
        "486ee1cba40b25f3ea8d6783f664eca88b7c12d01078984834d36834d9b2c802",
        251138448
    ),
    rfdetr!(
        "rfdetr-nano",
        "onnx-community/rfdetr_nano-ONNX",
        "eae21cee0687a91bcf9fa071605c48d7705d2d91",
        "RF-DETR nano",
        "RF-DETR nano, COCO 80 classes, 384x384 input; fastest RF-DETR",
        "9cbac6b11ce34a03034e4d5a24cfac5f18632fd6761d1311dd640232088d7fee",
        108074865
    ),
    rfdetr!(
        "rfdetr-small",
        "onnx-community/rfdetr_small-ONNX",
        "63463b68b200177d1fea7015f11f3cebb0ba4eeb",
        "RF-DETR small",
        "RF-DETR small, COCO 80 classes, 512x512 input; fast",
        "121cc1476a7b69d865ca4bdc2bca59a3239020d227e4c3f35a811bef81aeb7f1",
        114680416
    ),
    rfdetr!(
        "rfdetr-base",
        "onnx-community/rfdetr_base-ONNX",
        "7b18fa82ba7b1d6bc9446518cf2cc1d991097a3f",
        "RF-DETR base",
        "RF-DETR base, COCO 80 classes, 560x560 input; balanced accuracy and speed",
        "48b9502c04be895713efc0579c8904e69a97b3e03413a2f5aab1bdecda7e36ff",
        107773649
    ),
    rfdetr!(
        "rfdetr-medium",
        "onnx-community/rfdetr_medium-ONNX",
        "8553cefdd984bbaeb9d9ca8743d6a6a1c3e29dce",
        "RF-DETR medium",
        "RF-DETR medium, COCO 80 classes, 576x576 input; more accurate, slower",
        "768928acf337ac86d069c9d55a929e11e563c1e1d272a3ffc9366248d66b2654",
        121015411
    ),
    yolo5!(
        "delivery",
        "YOLOv5 delivery (vehicles, people, packages)",
        "1e66d71f3d5bbd6140e15716a43ca0e25fc5132af0a959c2696c11c1bf643e56",
        29571692,
        "17e0d52b7df884f834b9894395847319c0e872725327076df8684ae2045e962b",
        155
    ),
    yolo5!(
        "IPcam-animal",
        "YOLOv5 IP camera animals",
        "e62a9f8df5299476cf77e2351c711f7554e64b1200cca7756d7b58bbb9f32085",
        29616938,
        "9ca7e805c2c70013311688ebe1b339dfcb38f17216d5d499cc8195cc6b83ca61",
        227
    ),
    yolo5!(
        "ipcam-bird",
        "YOLOv5 IP camera birds",
        "265363bff2eda770463be1a1054eaaf722dec3703ec5b672a155fd1315a486be",
        84407925,
        "d79b8168621ca94221010b958bdebc08ce899ff68d4d03dc60937685f110b847",
        851
    ),
    yolo5!(
        "IPcam-combined",
        "YOLOv5 IP camera combined",
        "8f9f39828f979a4440151be2d32660e97f7a41e981af26002c5b638c4db9236a",
        29681724,
        "5224a2b0d32a0ea6ef9bb9a03ed54087280bffd0ee8228af3c46eb800b4a8315",
        308
    ),
    yolo5!(
        "IPcam-dark",
        "YOLOv5 IP camera night/dark scenes",
        "ca460c36ff279f0344756b3cadf936323fe01b28ef67b226cf00ca86b05808b7",
        29519768,
        "1e148765187f81e4fd931975810602eb5b7f976fb44ef4e679c578036e4b6558",
        114
    ),
    yolo5!(
        "IPcam-general",
        "YOLOv5 IP camera general purpose",
        "3b338be21f271a2924051fc237086a03c49daecd7e0fb2a8e97411ea5a33c6dc",
        29465778,
        "4070e087499ab1bf111efc58d98e6703b9457ce06ff0c22fb79d548074afb9c4",
        52
    ),
    yolo5!(
        "package",
        "YOLOv5 package detection",
        "b233f5a622a59f5a7ac7b48fbed79b5989ad6225a9569b3a8866a0bd75d9a9d7",
        29452878,
        "6187ff891bb16e442c0ce36593c04be118ab613299ea6060c4b1fd8c2855fe81",
        23
    ),
];

// ---------------------------------------------------------------------------------------------
// Models exported locally (Ultralytics YOLO, RF-DETR L, DEIMv2)
// ---------------------------------------------------------------------------------------------

/// How an exportable model is turned into `<name>.onnx` (`scripts/export_model.py --recipe`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExportRecipe {
    /// Ultralytics' exporter on an official `.pt` (YOLOv8/9/10/11/12/26). AGPL-3.0.
    Ultralytics,
    /// Hugging Face transformers `RfDetrForObjectDetection` + `torch.onnx` (static input,
    /// `pixel_values` -> `logits`, `pred_boxes`). Apache-2.0.
    RfDetr,
    /// The official DEIMv2 code (`DEIMV2_SRC_ID`) + `torch.onnx`, deploy mode (`images`,
    /// `orig_target_sizes` -> `labels`, `boxes`, `scores`).
    DeimV2,
}

impl ExportRecipe {
    /// `--recipe` value of the export script.
    pub const fn as_str(self) -> &'static str {
        match self {
            ExportRecipe::Ultralytics => "ultralytics",
            ExportRecipe::RfDetr => "rfdetr",
            ExportRecipe::DeimV2 => "deimv2",
        }
    }

    /// What runs on this machine, for confirmations and logs.
    pub const fn exporter(self) -> &'static str {
        match self {
            ExportRecipe::Ultralytics => "Ultralytics' exporter",
            ExportRecipe::RfDetr => "Hugging Face transformers and torch.onnx",
            ExportRecipe::DeimV2 => "the official DEIMv2 code (pinned commit) and torch.onnx",
        }
    }

    /// Shown before an export and in the Resources card.
    pub const fn notice(self) -> &'static str {
        match self {
            ExportRecipe::Ultralytics => ULTRALYTICS_NOTICE,
            ExportRecipe::RfDetr => RFDETR_NOTICE,
            ExportRecipe::DeimV2 => DEIMV2_NOTICE,
        }
    }
}

/// Ultralytics package version the export environment pins (`requirements-model-export.txt`).
pub const ULTRALYTICS_VERSION: &str = "8.4.163";
/// Ultralytics `assets` release with the YOLO weights (v8 to 26 are all in it).
pub const ULTRALYTICS_ASSETS_RELEASE: &str = "v8.4.0";
/// Version recorded for an Ultralytics export: exporter version + weights release (unchanged since
/// the YOLO26-only export, so earlier YOLO26 exports stay valid).
pub const ULTRALYTICS_EXPORT_VERSION: &str = "ultralytics-8.4.163+assets-v8.4.0";
/// License of the Ultralytics weights and exporter.
pub(crate) const ULTRALYTICS_LICENSE: &str = "AGPL-3.0";
const ULTRALYTICS_NOTICE: &str = "Ultralytics YOLO weights and the Ultralytics exporter are \
    licensed AGPL-3.0. Blue Onyx Prism does not redistribute them: on your request it downloads the \
    official weights from Ultralytics' GitHub release (SHA-256 pinned) and runs Ultralytics' \
    exporter on this machine in its own Python environment.";
/// transformers version of the export environment (RF-DETR).
pub const TRANSFORMERS_VERSION: &str = "5.17.0";
/// `Roboflow/rf-detr-large` commit of the RF-DETR L weights.
pub const RFDETR_LARGE_REVISION: &str = "f62f7dd5252b61097cbace33886045816dadbde9";
const RFDETR_NOTICE: &str = "RF-DETR (Apache-2.0) is published as safetensors only: on your \
    request Blue Onyx Prism downloads Roboflow's official weights from Hugging Face (SHA-256 \
    pinned) and exports them to ONNX on this machine with transformers and torch.onnx.";
/// The DEIMv2 source code resource (`Intellindust-AI-Lab/DEIMv2`, GitHub archive).
pub(crate) const DEIMV2_SRC_ID: &str = "tool:deimv2-src";
/// Install directory of [`DEIMV2_SRC_ID`] under the download root.
pub(crate) const DEIMV2_SRC_DEST: &str = "tools/deimv2-src";
/// DEIMv2 commit the source archive is pinned to (2026-08-14): the last one released under
/// Apache-2.0 (the repository switched to a non-commercial license on 2026-08-24 with docs-only
/// commits; the model code is the same).
pub(crate) const DEIMV2_COMMIT: &str = "add5bcdb499bf7b8a366bfeac1a47d3dc278de27";
const DEIMV2_NOTICE: &str = "DEIMv2 is exported on this machine from Intellindust's official \
    weights (Hugging Face, SHA-256 pinned, published 2025-10 under Apache-2.0) with the official \
    DEIMv2 code pinned to commit add5bcd, the last Apache-2.0 release; the repository has since \
    moved to a non-commercial license. The DINOv3 variants (S/M/L/X) also carry Meta's DINOv3 \
    License for the backbone.";

macro_rules! ultralytics {
    ($name:literal, $family:expr, $title:literal, $desc:literal, $sha:literal, $bytes:literal) => {
        Resource {
            id: concat!("model:", $name),
            kind: ResourceKind::ExportModel,
            version: ULTRALYTICS_EXPORT_VERSION,
            platform: None,
            parts: &[Part {
                url: concat!(
                    "https://github.com/ultralytics/assets/releases/download/v8.4.0/",
                    $name,
                    ".pt"
                ),
                sha256: $sha,
                size: $bytes,
                file_name: concat!($name, ".pt"),
                archive: None,
                layout: Layout::File,
            }],
            dest: "models",
            provides: Provides::Model {
                name: $name,
                family: $family,
            },
            title: $title,
            description: $desc,
            license: ULTRALYTICS_LICENSE,
        }
    };
}

/// YOLOv8 / v9 / 11 / 12: raw `[1, 84, 8400]` head, family `yolo8` (NMS in the service).
macro_rules! yolo_raw {
    ($($args:tt)*) => {
        ultralytics!($($args)*)
    };
}

/// A plain file of a Hugging Face repo at a commit.
macro_rules! hf_part {
    ($repo:expr, $rev:expr, $file:literal, $sha:literal, $size:literal) => {
        Part {
            url: concat!(
                "https://huggingface.co/",
                $repo,
                "/resolve/",
                $rev,
                "/",
                $file
            ),
            sha256: $sha,
            size: $size,
            file_name: $file,
            archive: None,
            layout: Layout::File,
        }
    };
}

/// A DEIMv2 COCO model (`Intellindust/DEIMv2_<variant>_COCO`): `model.safetensors` +
/// `config.json` (PyTorchModelHubMixin), family `rtdetr`.
macro_rules! deimv2 {
    ($name:literal, $variant:literal, $rev:literal, $title:literal, $desc:literal, $license:expr,
     $st_sha:literal, $st_size:literal, $cfg_sha:literal, $cfg_size:literal) => {
        Resource {
            id: concat!("model:", $name),
            kind: ResourceKind::ExportModel,
            version: concat!("deimv2-add5bcd+hf-", $rev),
            platform: None,
            parts: &[
                hf_part!(
                    concat!("Intellindust/DEIMv2_", $variant, "_COCO"),
                    $rev,
                    "model.safetensors",
                    $st_sha,
                    $st_size
                ),
                hf_part!(
                    concat!("Intellindust/DEIMv2_", $variant, "_COCO"),
                    $rev,
                    "config.json",
                    $cfg_sha,
                    $cfg_size
                ),
            ],
            dest: "models",
            provides: Provides::Model {
                name: $name,
                family: ModelFamilyKind::RtDetr,
            },
            title: $title,
            description: $desc,
            license: $license,
        }
    };
}

/// DEIMv2 is marked non-commercial / personal use: the pinned source (add5bcd) and the weights
/// predate the move to the non-commercial DEIMv2 License, but that license now claims the weights.
const DEIMV2_HGNET_LICENSE: &str = "Non-commercial / personal use only (DEIMv2 License)";
const DEIMV2_DINOV3_LICENSE: &str =
    "Non-commercial / personal use only (DEIMv2 License) + DINOv3 License (backbone)";

/// Models exported on this machine on request ([`crate::resources::export`]): the parts are the official
/// weights (and configs); the export writes `models/<name>.onnx` + `models/<name>.yaml` (80 COCO
/// names). All COCO 80 classes. YOLOv10 and YOLO26 are end-to-end (`[1,300,6]`, family
/// `yolo26`); the other YOLOs are raw heads (family `yolo8`). Never downloaded or exported
/// automatically.
pub static EXPORT_MODELS: &[Resource] = &[
    // YOLOv8
    yolo_raw!(
        "yolov8n",
        ModelFamilyKind::Yolo8,
        "YOLOv8 nano",
        "YOLOv8 nano, COCO 80 classes; fast and small, a good CPU/iGPU baseline",
        "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
        6549796
    ),
    yolo_raw!(
        "yolov8s",
        ModelFamilyKind::Yolo8,
        "YOLOv8 small",
        "YOLOv8 small, COCO 80 classes; more accurate than nano, still light",
        "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a",
        22588772
    ),
    yolo_raw!(
        "yolov8m",
        ModelFamilyKind::Yolo8,
        "YOLOv8 medium",
        "YOLOv8 medium, COCO 80 classes; accurate, best with a GPU",
        "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5",
        52136884
    ),
    yolo_raw!(
        "yolov8l",
        ModelFamilyKind::Yolo8,
        "YOLOv8 large",
        "YOLOv8 large, COCO 80 classes; slow without a GPU",
        "64c9115303f6a25575f82200d1b22ec409fa6bd7d08d0313884fc20d919478cd",
        87792836
    ),
    yolo_raw!(
        "yolov8x",
        ModelFamilyKind::Yolo8,
        "YOLOv8 extra large",
        "YOLOv8 extra large, COCO 80 classes; needs a fast GPU",
        "3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8",
        136890692
    ),
    // YOLOv9
    yolo_raw!(
        "yolov9t",
        ModelFamilyKind::Yolo8,
        "YOLOv9 tiny",
        "YOLOv9 tiny, COCO 80 classes; smallest YOLOv9",
        "410e3caca789aa8fec19971c05577616cbb963a416bcd0228171b2dff7f718f7",
        4969803
    ),
    yolo_raw!(
        "yolov9s",
        ModelFamilyKind::Yolo8,
        "YOLOv9 small",
        "YOLOv9 small, COCO 80 classes; light and accurate for its size",
        "67731fc88bb6782fbb4f32703a2cac052acd65148b65c9e67161fe13536c2ffc",
        15389259
    ),
    yolo_raw!(
        "yolov9m",
        ModelFamilyKind::Yolo8,
        "YOLOv9 medium",
        "YOLOv9 medium, COCO 80 classes; accurate, best with a GPU",
        "be7ac91362bd81dfa901077706d634e571c1ff7f3886b3aac6a0e97eff2f5440",
        40993492
    ),
    yolo_raw!(
        "yolov9c",
        ModelFamilyKind::Yolo8,
        "YOLOv9 compact",
        "YOLOv9 compact (C), COCO 80 classes; large, needs a GPU",
        "876eb84f515d40c34a3b111f8fc1077d3aee59d3a243afd1cc5b77d520f237c7",
        51794840
    ),
    yolo_raw!(
        "yolov9e",
        ModelFamilyKind::Yolo8,
        "YOLOv9 extended",
        "YOLOv9 extended (E), COCO 80 classes; most accurate YOLOv9, needs a fast GPU",
        "0d2270e0c794bdaa059738d32c1781cdec4857b3931d5503e13ae92e8517e7bd",
        117521038
    ),
    // YOLOv10 (end-to-end, no NMS)
    ultralytics!(
        "yolov10n",
        ModelFamilyKind::Yolo26,
        "YOLOv10 nano",
        "YOLOv10 nano, COCO 80 classes, NMS-free; fast",
        "11287ed0735678e7ba1ac2a9b3098c049155b3fde123992e724c1264bcc16b6f",
        5860383
    ),
    ultralytics!(
        "yolov10s",
        ModelFamilyKind::Yolo26,
        "YOLOv10 small",
        "YOLOv10 small, COCO 80 classes, NMS-free; more accurate, still light",
        "65fa8332c38a7189597a268897bce6c1026f9499711b9761fb0c2c639b91d4d6",
        16623111
    ),
    ultralytics!(
        "yolov10m",
        ModelFamilyKind::Yolo26,
        "YOLOv10 medium",
        "YOLOv10 medium, COCO 80 classes, NMS-free; accurate, best with a GPU",
        "6dc78f7a88591cec1e8716b8f5c7e3aefa9206684f025d202be34439ccb329a0",
        33643667
    ),
    ultralytics!(
        "yolov10b",
        ModelFamilyKind::Yolo26,
        "YOLOv10 balanced",
        "YOLOv10 balanced (B), COCO 80 classes, NMS-free; needs a GPU",
        "31ef0baecfa8010ebb4b53b0624836f30d73702b04710c23945782dba5d258c2",
        41679663
    ),
    ultralytics!(
        "yolov10l",
        ModelFamilyKind::Yolo26,
        "YOLOv10 large",
        "YOLOv10 large, COCO 80 classes, NMS-free; needs a GPU",
        "d0685e1e597f73aba8e56bc9651b6bb939c5b2f201dff5de0b9ef2a6207ae301",
        52425230
    ),
    ultralytics!(
        "yolov10x",
        ModelFamilyKind::Yolo26,
        "YOLOv10 extra large",
        "YOLOv10 extra large, COCO 80 classes, NMS-free; needs a fast GPU",
        "605982e0b2c2ff070351afc8df3ecb38c54cfd7f133813acb366f454eb79b9a6",
        64395854
    ),
    // YOLO11
    yolo_raw!(
        "yolo11n",
        ModelFamilyKind::Yolo8,
        "YOLO11 nano",
        "YOLO11 nano, COCO 80 classes; fast, good on CPUs and iGPUs",
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        5613764
    ),
    yolo_raw!(
        "yolo11s",
        ModelFamilyKind::Yolo8,
        "YOLO11 small",
        "YOLO11 small, COCO 80 classes; more accurate, about 2x nano's time",
        "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
        19313732
    ),
    yolo_raw!(
        "yolo11m",
        ModelFamilyKind::Yolo8,
        "YOLO11 medium",
        "YOLO11 medium, COCO 80 classes; accurate, best with a GPU",
        "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
        40684120
    ),
    yolo_raw!(
        "yolo11l",
        ModelFamilyKind::Yolo8,
        "YOLO11 large",
        "YOLO11 large, COCO 80 classes; slow without a GPU",
        "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f",
        51387343
    ),
    yolo_raw!(
        "yolo11x",
        ModelFamilyKind::Yolo8,
        "YOLO11 extra large",
        "YOLO11 extra large, COCO 80 classes; needs a fast GPU",
        "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24",
        114636239
    ),
    // YOLO12
    yolo_raw!(
        "yolo12n",
        ModelFamilyKind::Yolo8,
        "YOLO12 nano",
        "YOLO12 nano, COCO 80 classes (attention-based); fast",
        "419ff3dca37d69bacc93a50fa0c186a1c6f9fe62fae0f108b0872829689e9ca6",
        5595063
    ),
    yolo_raw!(
        "yolo12s",
        ModelFamilyKind::Yolo8,
        "YOLO12 small",
        "YOLO12 small, COCO 80 classes (attention-based); more accurate",
        "e915c2c4286e3f6f8610ef106fa3f94a7b8c19b30eccede5887e22c33ef75f58",
        19006455
    ),
    yolo_raw!(
        "yolo12m",
        ModelFamilyKind::Yolo8,
        "YOLO12 medium",
        "YOLO12 medium, COCO 80 classes (attention-based); best with a GPU",
        "4c6d179786eddf6134ee469ae2f4ce04cbe4e9d1a47d6b669d9cd6b9c6c513d8",
        40904955
    ),
    yolo_raw!(
        "yolo12l",
        ModelFamilyKind::Yolo8,
        "YOLO12 large",
        "YOLO12 large, COCO 80 classes (attention-based); needs a GPU",
        "0babd8dc8f775bb64bb052debdff3d8b9e9b57efa9d7bfa11c84bb82c3fec336",
        53699086
    ),
    yolo_raw!(
        "yolo12x",
        ModelFamilyKind::Yolo8,
        "YOLO12 extra large",
        "YOLO12 extra large, COCO 80 classes (attention-based); needs a fast GPU",
        "682ce8dadee004dbe964950f1bf3eda451671815a6ed62db80b398916b9b7c6f",
        119322638
    ),
    // YOLO26 (end-to-end, no NMS). Speeds: ONNX Runtime CoreML on an Apple M1.
    ultralytics!(
        "yolo26n",
        ModelFamilyKind::Yolo26,
        "YOLO26 nano",
        "YOLO26 nano, COCO 80 classes; fastest (about 15 ms on an M1 with CoreML), good on CPUs and iGPUs",
        "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        5544453
    ),
    ultralytics!(
        "yolo26s",
        ModelFamilyKind::Yolo26,
        "YOLO26 small",
        "YOLO26 small, COCO 80 classes; more accurate, about 2x nano's time (29 ms on an M1 with CoreML)",
        "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
        20422725
    ),
    ultralytics!(
        "yolo26m",
        ModelFamilyKind::Yolo26,
        "YOLO26 medium",
        "YOLO26 medium, COCO 80 classes; accurate, best with a GPU",
        "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
        44255705
    ),
    ultralytics!(
        "yolo26l",
        ModelFamilyKind::Yolo26,
        "YOLO26 large",
        "YOLO26 large, COCO 80 classes; slower, needs a GPU for camera streams",
        "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26",
        53211173
    ),
    ultralytics!(
        "yolo26x",
        ModelFamilyKind::Yolo26,
        "YOLO26 extra large",
        "YOLO26 extra large, COCO 80 classes; most accurate and slowest, needs a fast GPU",
        "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92",
        118667365
    ),
    // RF-DETR L (current release, DINOv2 windowed backbone, 704x704), family `rfdetr`.
    Resource {
        id: "model:rfdetr-large",
        kind: ResourceKind::ExportModel,
        version: "transformers-5.17.0+rf-detr-large-f62f7dd5252b61097cbace33886045816dadbde9",
        platform: None,
        parts: &[
            hf_part!(
                "Roboflow/rf-detr-large",
                "f62f7dd5252b61097cbace33886045816dadbde9",
                "model.safetensors",
                "1ec604598aea748627eb5755a9989f3c80839fc7fb977a91bfba107ee76fcd7d",
                135795376
            ),
            hf_part!(
                "Roboflow/rf-detr-large",
                "f62f7dd5252b61097cbace33886045816dadbde9",
                "config.json",
                "6630951d97cc8dd2f32da43df7c79af153b032f6df196bbe9bf9e47bd7be0c0c",
                5771
            ),
        ],
        dest: "models",
        provides: Provides::Model {
            name: "rfdetr-large",
            family: ModelFamilyKind::RfDetr,
        },
        title: "RF-DETR large",
        description: "RF-DETR large (current, 33.9M params), COCO 80 classes, 704x704 input; most accurate RF-DETR, needs a GPU",
        license: "Apache-2.0",
    },
    // DEIMv2 (HGNetv2 backbones: ImageNet-free input; DINOv3: normalization baked into the graph).
    deimv2!(
        "deimv2-atto",
        "HGNetv2_ATTO",
        "21fbe0e24abeb35ed216dbfd8a32855dbc03214f",
        "DEIMv2 atto",
        "DEIMv2 atto (0.5M params), COCO 80 classes, 320x320 input; tiny, for very weak CPUs",
        DEIMV2_HGNET_LICENSE,
        "774dd877ef7b90e10d4136f31555f34627a62dc00a0d6908d9b1a48574002730",
        2120688,
        "51bcdedb67ac127b6f955c1f919c7836db3eb358465e2c7057bba8c270368081",
        822
    ),
    deimv2!(
        "deimv2-femto",
        "HGNetv2_FEMTO",
        "ccbd03616ce5373c32622c605fb29bbb7dec4b52",
        "DEIMv2 femto",
        "DEIMv2 femto (1.0M params), COCO 80 classes, 416x416 input; very small",
        DEIMV2_HGNET_LICENSE,
        "db7026cc3a7c308934a8cf3fa8244f9967078d7cc8ec20da8e5250ef8842e75f",
        4033545,
        "b37530572fb615601f18f7876e1619d00b73cf9e71c3a25c0b79ba03c7eb3717",
        823
    ),
    deimv2!(
        "deimv2-pico",
        "HGNetv2_PICO",
        "849bb61f94f3d42a8977cabb38b998fa91fed44a",
        "DEIMv2 pico",
        "DEIMv2 pico (1.5M params), COCO 80 classes, 640x640 input; small and fast",
        DEIMV2_HGNET_LICENSE,
        "788b3a42297333ae3d47c337a89d859187f9000a7303f748773e88db8a454e0f",
        6293772,
        "ae9c327e8d281e626e324660d8c71480abe8ac6c91dd4fe57645480d82e6b84f",
        826
    ),
    deimv2!(
        "deimv2-n",
        "HGNetv2_N",
        "47f40396e40453719386cd5c238a57aae954e83f",
        "DEIMv2 nano",
        "DEIMv2 nano (3.6M params), COCO 80 classes, 640x640 input; fast, good on CPUs and iGPUs",
        DEIMV2_HGNET_LICENSE,
        "0de14388a703d86c95200589b32a4497b0b00dceb1d2d381c6a14fbd958856cb",
        14574360,
        "6eb57450791c8ada4b76811ad4fbcf040a58911ecf3dff5a785487389d4443f8",
        1326
    ),
    deimv2!(
        "deimv2-s",
        "DINOv3_S",
        "cf0540f3f319bb8ecbe132624358a35a7beb86d5",
        "DEIMv2 small",
        "DEIMv2 small (DINOv3-distilled ViT, 9.7M params), COCO 80 classes; accurate for its size",
        DEIMV2_DINOV3_LICENSE,
        "fe545b150766b8761a696f7b1d92ea138fafe8e02a7398823e0c62d2a5867956",
        39386484,
        "c815791122b6be862d5c5902b3557771bda640acd0ca21ca9999c540eb6e6853",
        1321
    ),
    deimv2!(
        "deimv2-m",
        "DINOv3_M",
        "c5db2fc6f4d9e59635cfc2effaf097bc9a293beb",
        "DEIMv2 medium",
        "DEIMv2 medium (DINOv3-distilled ViT, 18.1M params), COCO 80 classes; accurate, best with a GPU",
        DEIMV2_DINOV3_LICENSE,
        "f5d6beb9c032b7cdd24da6eebf6966da6c7d2d49b312ac6c0198f112170c0af6",
        73752660,
        "468aa716b1cec24298dfdba4c07c07eab1ff4b6e7a3a64c7396a367a93d7a67b",
        1322
    ),
    deimv2!(
        "deimv2-l",
        "DINOv3_L",
        "6ce90ce18c14271c8ce145c35626d9a6544f881b",
        "DEIMv2 large",
        "DEIMv2 large (DINOv3 ViT-S, 32.2M params), COCO 80 classes; needs a GPU",
        DEIMV2_DINOV3_LICENSE,
        "c0d0f45fa62b785126db5b40f5530866fecaab2d340cea812d2ba5f0f5646163",
        130596044,
        "9800c67c83501a1f5159b5b372c7f96b367b329d3713ca6d7309c86b2e7a0319",
        1373
    ),
    deimv2!(
        "deimv2-x",
        "DINOv3_X",
        "b1d81486dce5686b2afcd38a231d086d142c996d",
        "DEIMv2 extra large",
        "DEIMv2 extra large (DINOv3 ViT-S+, 50.3M params), COCO 80 classes; most accurate DEIMv2, needs a fast GPU",
        DEIMV2_DINOV3_LICENSE,
        "c15e1860118e6841f8f803219507dbfb630c2aa0164ab723c339b94e554bae7f",
        205313972,
        "456e2f740828eb52bc3dcc789de28af67f366cdf2b415f2338bc3b4e2f28fae4",
        1382
    ),
];

/// The recipe of an exportable model (keyed on its name; every [`EXPORT_MODELS`] entry has one,
/// checked by a test).
pub fn export_recipe(res: &Resource) -> Option<ExportRecipe> {
    if res.kind != ResourceKind::ExportModel {
        return None;
    }
    let name = res.model_name()?;
    if name.starts_with("yolo") {
        Some(ExportRecipe::Ultralytics)
    } else if name.starts_with("rfdetr") {
        Some(ExportRecipe::RfDetr)
    } else if name.starts_with("deimv2") {
        Some(ExportRecipe::DeimV2)
    } else {
        None
    }
}

/// Square input side an export of `res` uses (`--imgsz`): [`MODEL_INPUT_SIZES`], else 640.
pub(crate) fn export_input_size(res: &Resource) -> u32 {
    res.model_name()
        .and_then(model_input_size)
        .unwrap_or(crate::model::DEFAULT_INPUT_SIZE)
}

/// Other resources an export of `res` needs besides its weights (the DEIMv2 source code).
pub(crate) fn export_extras(res: &Resource) -> Vec<&'static Resource> {
    match export_recipe(res) {
        Some(ExportRecipe::DeimV2) => deimv2_source().into_iter().collect(),
        _ => Vec::new(),
    }
}

/// What setting up the model export toolchain costs on one platform (uv, Python 3.11 and the
/// pinned packages; the weights are extra). Download sizes are the sums of the locked wheels (see
/// `scripts/model-export/gen_locks.sh`; macOS 334 MB, Linux x64 443 MB, Linux arm64 379 MB, Windows
/// 321 MB) plus uv and Python; disk is `tools/` afterwards (measured on macOS arm64; the others
/// scaled by their wheel sizes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ToolchainEstimate {
    pub platform: Platform,
    /// Bytes downloaded once (uv archive + Python + wheels).
    pub download: u64,
    /// Bytes on disk afterwards (`tools/`).
    pub disk: u64,
}

/// Platforms the model export supports (PyTorch 2.14 CPU wheels exist), with their costs.
pub(crate) const TOOLCHAIN_ESTIMATES: &[ToolchainEstimate] = &[
    ToolchainEstimate {
        platform: Platform::MACOS_ARM64,
        download: 375_000_000,
        disk: 1_300_000_000,
    },
    ToolchainEstimate {
        platform: Platform::LINUX_X64,
        download: 495_000_000,
        disk: 1_700_000_000,
    },
    ToolchainEstimate {
        platform: Platform::LINUX_ARM64,
        download: 425_000_000,
        disk: 1_500_000_000,
    },
    ToolchainEstimate {
        platform: Platform::WINDOWS_X64,
        download: 385_000_000,
        disk: 1_400_000_000,
    },
];

/// The export toolchain estimate for `(os, arch)`; None = models cannot be exported there.
pub fn toolchain_estimate(os: &str, arch: &str) -> Option<&'static ToolchainEstimate> {
    TOOLCHAIN_ESTIMATES
        .iter()
        .find(|e| e.platform.matches(os, arch))
}

/// Exportable model by name or file name (`yolo26s`, `YOLO26S.onnx`, `yolo26s.xml`,
/// `model:deimv2-n`).
pub fn export_model(name: &str) -> Option<&'static Resource> {
    let n = name.trim();
    let stem = std::path::Path::new(n)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(n);
    let stem = strip_prefix_ci(stem, MODEL_ID_PREFIX).unwrap_or(stem);
    EXPORT_MODELS
        .iter()
        .find(|r| r.model_name().is_some_and(|m| m.eq_ignore_ascii_case(stem)))
}

/// File names an export of `res` writes into the models directory: `(<name>.onnx, <name>.yaml)`.
pub fn export_files(res: &Resource) -> Option<(String, String)> {
    (res.kind == ResourceKind::ExportModel)
        .then(|| res.model_name())
        .flatten()
        .map(|n| (format!("{n}.onnx"), format!("{n}.yaml")))
}
