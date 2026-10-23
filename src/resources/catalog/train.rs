//! Training pins: the model families and sizes offered for training ([`TrainFamily`]), the base
//! weights each one starts from ([`base_weights`]: the Ultralytics `.pt` pins of
//! [`EXPORT_MODELS`] reused, RF-DETR / RT-DETRv2 / D-FINE checkpoints in
//! [`TRAIN_BASE_WEIGHTS`]) and the optional public night subset ([`lila_night`], a manifest of
//! image URLs + SHA-256 like the benchmark sets). Pure data; only [`crate::training`] uses it,
//! when the user starts a training run. Nothing here is downloaded automatically.
//!
//! Where the hashes come from: RF-DETR checkpoints downloaded from Roboflow's bucket and hashed
//! (their MD5 matches the `rfdetr` 1.11.2 package's own table); Hugging Face LFS `oid`s for the
//! RT-DETRv2 / D-FINE safetensors and the downloaded, hashed `config.json` /
//! `preprocessor_config.json` at the pinned commits.

use super::*;

/// Who trains a family (`scripts/train_model.py --trainer`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Trainer {
    /// `ultralytics` (`YOLO(<base>.pt).train(...)`).
    Ultralytics,
    /// The `rfdetr` package (`RFDETRNano(...).train(...)`, COCO layout).
    RfDetr,
    /// Hugging Face transformers' `Trainer` (RT-DETRv2, D-FINE).
    Transformers,
}

impl Trainer {
    pub const fn as_str(self) -> &'static str {
        match self {
            Trainer::Ultralytics => "ultralytics",
            Trainer::RfDetr => "rfdetr",
            Trainer::Transformers => "transformers",
        }
    }
}

/// A family that can be fine-tuned. DEIMv2 is not offered (its license now restricts the
/// weights), nor RF-DETR XL / 2XL (Roboflow's PML license, non-commercial).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainFamily {
    Yolo26,
    Yolo11,
    Yolov8,
    Rfdetr,
    Rtdetrv2,
    Dfine,
}

/// Sizes of the Ultralytics families.
const YOLO_SIZES: &[&str] = &["n", "s", "m", "l", "x"];
/// Sizes of RF-DETR (Apache-2.0 ones only).
const RFDETR_SIZES: &[&str] = &["nano", "small", "medium", "base", "large"];
/// Sizes of RT-DETRv2 and D-FINE.
const DETR_SIZES: &[&str] = &["s", "m", "l", "x"];

/// Licence of a model trained from Ultralytics weights with Ultralytics' trainer.
pub const AGPL: &str = "AGPL-3.0";
/// Licence of a model trained from the Apache-2.0 families.
pub const APACHE: &str = "Apache-2.0";

impl TrainFamily {
    pub const ALL: [TrainFamily; 6] = [
        TrainFamily::Yolo26,
        TrainFamily::Yolo11,
        TrainFamily::Yolov8,
        TrainFamily::Rfdetr,
        TrainFamily::Rtdetrv2,
        TrainFamily::Dfine,
    ];

    /// `--family` of the train script and the CLI.
    pub const fn as_str(self) -> &'static str {
        match self {
            TrainFamily::Yolo26 => "yolo26",
            TrainFamily::Yolo11 => "yolo11",
            TrainFamily::Yolov8 => "yolov8",
            TrainFamily::Rfdetr => "rfdetr",
            TrainFamily::Rtdetrv2 => "rtdetrv2",
            TrainFamily::Dfine => "dfine",
        }
    }

    /// Case-insensitive; accepts `rt-detrv2`, `d-fine`, `rf-detr`.
    pub fn parse(s: &str) -> Option<Self> {
        let n = s.trim().to_ascii_lowercase().replace('-', "");
        TrainFamily::ALL.into_iter().find(|f| f.as_str() == n)
    }

    pub const fn title(self) -> &'static str {
        match self {
            TrainFamily::Yolo26 => "YOLO26",
            TrainFamily::Yolo11 => "YOLO11",
            TrainFamily::Yolov8 => "YOLOv8",
            TrainFamily::Rfdetr => "RF-DETR",
            TrainFamily::Rtdetrv2 => "RT-DETRv2",
            TrainFamily::Dfine => "D-FINE",
        }
    }

    pub const fn trainer(self) -> Trainer {
        match self {
            TrainFamily::Yolo26 | TrainFamily::Yolo11 | TrainFamily::Yolov8 => Trainer::Ultralytics,
            TrainFamily::Rfdetr => Trainer::RfDetr,
            TrainFamily::Rtdetrv2 | TrainFamily::Dfine => Trainer::Transformers,
        }
    }

    pub const fn sizes(self) -> &'static [&'static str] {
        match self {
            TrainFamily::Yolo26 | TrainFamily::Yolo11 | TrainFamily::Yolov8 => YOLO_SIZES,
            TrainFamily::Rfdetr => RFDETR_SIZES,
            TrainFamily::Rtdetrv2 | TrainFamily::Dfine => DETR_SIZES,
        }
    }

    /// The size spelled as this family spells it (`small` -> `s` for YOLO, `s` -> `small` for
    /// RF-DETR), or None.
    pub fn size(self, s: &str) -> Option<&'static str> {
        let s = s.trim().to_ascii_lowercase();
        let long = [
            ("n", "nano"),
            ("s", "small"),
            ("m", "medium"),
            ("b", "base"),
            ("l", "large"),
            ("x", "xlarge"),
        ];
        let sizes = self.sizes();
        sizes.iter().copied().find(|z| *z == s).or_else(|| {
            long.iter()
                .find(|(short, full)| *short == s || *full == s)
                .and_then(|(short, full)| {
                    sizes
                        .iter()
                        .copied()
                        .find(|z| z == short || z == full || (*full == "xlarge" && *z == "x"))
                })
        })
    }

    /// Licence of the trained model.
    pub const fn license(self) -> &'static str {
        match self.trainer() {
            Trainer::Ultralytics => AGPL,
            _ => APACHE,
        }
    }

    /// The service family of the exported model.
    pub const fn service_family(self) -> ModelFamilyKind {
        match self {
            TrainFamily::Yolo26 => ModelFamilyKind::Yolo26,
            TrainFamily::Yolo11 | TrainFamily::Yolov8 => ModelFamilyKind::Yolo8,
            TrainFamily::Rfdetr => ModelFamilyKind::RfDetr,
            TrainFamily::Rtdetrv2 | TrainFamily::Dfine => ModelFamilyKind::Detr,
        }
    }

    /// Default square input size of `size` (RF-DETR: the size's pretraining resolution).
    pub fn default_imgsz(self, size: &str) -> u32 {
        match (self, size) {
            (TrainFamily::Rfdetr, "nano") => 384,
            (TrainFamily::Rfdetr, "small") => 512,
            (TrainFamily::Rfdetr, "medium") => 576,
            (TrainFamily::Rfdetr, "base") => 560,
            (TrainFamily::Rfdetr, "large") => 704,
            _ => 640,
        }
    }

    /// The input size must be a multiple of this (RF-DETR: patch size x windows).
    pub fn imgsz_step(self, size: &str) -> u32 {
        match (self, size) {
            (TrainFamily::Rfdetr, "base") => 56,
            _ => 32,
        }
    }

    /// Whether a 1-epoch smoke run passed on Apple MPS (PyTorch 2.14, macOS 15, M1; see
    /// docs/TRAINING.md). D-FINE fails in the backward pass ("mat2 must be a matrix") on MPS,
    /// also with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
    pub const fn mps_ok(self) -> bool {
        !matches!(self, TrainFamily::Dfine)
    }

    /// Why a family is not offered on MPS, when it is not.
    pub const fn mps_note(self) -> Option<&'static str> {
        match self {
            TrainFamily::Dfine => Some(
                "D-FINE training fails on Apple MPS (PyTorch 2.14: \"mat2 must be a matrix\" in \
                 the backward pass); train it on an NVIDIA GPU",
            ),
            _ => None,
        }
    }

    /// Shown before a run: what is downloaded and what licence the result carries.
    pub const fn notice(self) -> &'static str {
        match self.trainer() {
            Trainer::Ultralytics => {
                "Ultralytics YOLO weights and trainer are AGPL-3.0, so is a model trained from \
                 them: keep it for your own use, or publish it (and the code that serves it) under \
                 AGPL-3.0."
            }
            Trainer::RfDetr => {
                "RF-DETR (Roboflow, Apache-2.0): the result is Apache-2.0. Base weights from \
                 Roboflow's bucket, SHA-256 pinned."
            }
            Trainer::Transformers => {
                "Apache-2.0 base weights from Hugging Face (SHA-256 pinned, COCO-only \
                 checkpoints; the D-FINE Objects365 checkpoints are not offered), trained with \
                 transformers; the result is Apache-2.0."
            }
        }
    }

    /// Name of the base model in the catalog / benchmark (`yolo26n`, `rfdetr-nano`,
    /// `rt-detrv2-s`, `dfine-s`), for the latency hint from `benchmark.json`.
    pub fn base_name(self, size: &str) -> String {
        match self {
            TrainFamily::Yolo26 => format!("yolo26{size}"),
            TrainFamily::Yolo11 => format!("yolo11{size}"),
            TrainFamily::Yolov8 => format!("yolov8{size}"),
            TrainFamily::Rfdetr => format!("rfdetr-{size}"),
            TrainFamily::Rtdetrv2 => format!("rt-detrv2-{size}"),
            TrainFamily::Dfine => format!("dfine-{size}"),
        }
    }
}

impl std::fmt::Display for TrainFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Id prefix of the training base weights.
pub const TRAIN_BASE_PREFIX: &str = "train-base:";
/// Install directory of [`TRAIN_BASE_WEIGHTS`] under the download root (`<dest>/<family>-<size>`).
pub const TRAIN_WEIGHTS_DEST: &str = "tools/train-weights";

/// A plain file of a Hugging Face repo at a commit.
macro_rules! hf_train_part {
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

/// An RF-DETR COCO checkpoint from Roboflow's bucket (the `rfdetr` package's own pins).
macro_rules! rfdetr_base {
    ($id:literal, $size:literal, $url:literal, $file:literal, $sha:literal, $bytes:literal, $title:literal) => {
        Resource {
            id: concat!("train-base:rfdetr-", $size),
            kind: ResourceKind::ExportModel,
            version: concat!("rfdetr-1.11.2+", $id),
            platform: None,
            parts: &[Part {
                url: $url,
                sha256: $sha,
                size: $bytes,
                file_name: $file,
                archive: None,
                layout: Layout::File,
            }],
            dest: TRAIN_WEIGHTS_DEST,
            provides: Provides::Tool {
                name: concat!("train-base-rfdetr-", $size),
            },
            title: $title,
            description: "RF-DETR COCO checkpoint (training base; Apache-2.0)",
            license: APACHE,
        }
    };
}

/// A Hugging Face transformers checkpoint at a commit: `config.json`, `model.safetensors`,
/// `preprocessor_config.json`.
macro_rules! hf_base {
    ($id:literal, $repo:literal, $rev:literal, $title:literal,
     $cfg_sha:literal, $cfg_size:literal, $st_sha:literal, $st_size:literal, $pp_sha:literal, $pp_size:literal) => {
        Resource {
            id: concat!("train-base:", $id),
            kind: ResourceKind::ExportModel,
            version: concat!("hf-", $rev),
            platform: None,
            parts: &[
                hf_train_part!($repo, $rev, "config.json", $cfg_sha, $cfg_size),
                hf_train_part!($repo, $rev, "model.safetensors", $st_sha, $st_size),
                hf_train_part!($repo, $rev, "preprocessor_config.json", $pp_sha, $pp_size),
            ],
            dest: TRAIN_WEIGHTS_DEST,
            provides: Provides::Tool {
                name: concat!("train-base-", $id),
            },
            title: $title,
            description: concat!($repo, " (training base; Apache-2.0)"),
            license: APACHE,
        }
    };
}

/// Base weights of the families that have no export pin to reuse. The RF-DETR checkpoints carry
/// their training state (optimizer), hence the size.
pub static TRAIN_BASE_WEIGHTS: &[Resource] = &[
    rfdetr_base!(
        "nano_coco",
        "nano",
        "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
        "rf-detr-nano.pth",
        "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d",
        366287238,
        "RF-DETR nano COCO checkpoint"
    ),
    rfdetr_base!(
        "small_coco",
        "small",
        "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
        "rf-detr-small.pth",
        "d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33",
        386045550,
        "RF-DETR small COCO checkpoint"
    ),
    rfdetr_base!(
        "medium_coco",
        "medium",
        "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
        "rf-detr-medium.pth",
        "749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8",
        404992918,
        "RF-DETR medium COCO checkpoint"
    ),
    rfdetr_base!(
        "base_coco",
        "base",
        "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
        "rf-detr-base.pth",
        "d8f70210e425a4a4234d547737f57500bcc4ac24a333b99e33d9d5a371e0b80f",
        372578043,
        "RF-DETR base COCO checkpoint"
    ),
    rfdetr_base!(
        "large_2026",
        "large",
        "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
        "rf-detr-large-2026.pth",
        "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38",
        135954129,
        "RF-DETR large COCO checkpoint"
    ),
    hf_base!(
        "rtdetrv2-s",
        "PekingU/rtdetr_v2_r18vd",
        "5650961749fa93567c0d46fc7f43ea4f9e914107",
        "RT-DETRv2 S (ResNet-18) COCO checkpoint",
        "ed051ec77cb41c5d9d5e3af21a979b1e890599dfd68434d7636ff82ded4c1527",
        5437,
        "d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115",
        80904640,
        "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",
        444
    ),
    hf_base!(
        "rtdetrv2-m",
        "PekingU/rtdetr_v2_r34vd",
        "5d60d8fb5eb2a3ee5575bf200ecbce4e2cfa420c",
        "RT-DETRv2 M (ResNet-34) COCO checkpoint",
        "9315791fc96f2e1014d98c44485c1aeaea282f6093f29dc2fb249af3fd57ef75",
        5379,
        "d67528a6c4bbaece03d65feefbba94052894aa6520aa1f65c083cd0f7b92d612",
        125979400,
        "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",
        444
    ),
    hf_base!(
        "rtdetrv2-l",
        "PekingU/rtdetr_v2_r50vd",
        "282494075698cab9faa1096ae26856890030c817",
        "RT-DETRv2 L (ResNet-50) COCO checkpoint",
        "d754c04c9a4c6d015d8393fa2dae4c63f4a9f0ba173c3d1a66e55cff59d8f060",
        5283,
        "3331d977dbc0c7a6cdae9ec0b0b6ad156eb6720d65b7cf0fa710dcc541d88d71",
        172176856,
        "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",
        444
    ),
    hf_base!(
        "rtdetrv2-x",
        "PekingU/rtdetr_v2_r101vd",
        "2c5dbbd2d4d8c8814827a3b42737ba1afce3cf2a",
        "RT-DETRv2 X (ResNet-101) COCO checkpoint",
        "6d2fe3e6b5160a044d68a12435a4deac3ef7b0bebaca65a5beb1da10ddae377f",
        5342,
        "fa2d79aa50064eb2d41abdc1acaa5ca20fc9d95347851c93ab6aad30a8974ee0",
        307332000,
        "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",
        444
    ),
    hf_base!(
        "dfine-s",
        "ustc-community/dfine-small-coco",
        "f79e65b5fbb33ceb9d3ebba042955d7410c608f8",
        "D-FINE S COCO checkpoint",
        "9f206c4c935c67c957af2f15c3333b4ff224e72b1881bc7bbde9f873edf4448e",
        6865,
        "a144baff3a27bfd538fe37b11fa8d4fca5744b8c054b5301e7c699a1379cd18f",
        41497620,
        "7e9e2245806aa3ef068a5321f5eebee3116fa4cec560173231954ffaca930eca",
        470
    ),
    hf_base!(
        "dfine-m",
        "ustc-community/dfine-medium-coco",
        "4ab9f5e466432f4fcf9a9a023da6aa5ecc9c9829",
        "D-FINE M COCO checkpoint",
        "751deb25038ced3c7bbc8515914d71a0b4a5485486cd7c7a119c0adee209175e",
        6967,
        "5a53154fb87af91ad1ca3081599f21423a5681c46b613e15f2ff556837b02634",
        78699728,
        "7e9e2245806aa3ef068a5321f5eebee3116fa4cec560173231954ffaca930eca",
        470
    ),
    hf_base!(
        "dfine-l",
        "ustc-community/dfine-large-coco",
        "aa8f8211545e4cff5adf52ea512f9ac74a9824ec",
        "D-FINE L COCO checkpoint",
        "f2c0253fc97a75269dc4016e373122f87f7efa7d675d4ad0dedea6f050136b8d",
        6959,
        "4e9d77ba937ab6eb8537b183c1c05372262daaa552e7682d721417042e24a8dc",
        125618504,
        "7e9e2245806aa3ef068a5321f5eebee3116fa4cec560173231954ffaca930eca",
        470
    ),
    hf_base!(
        "dfine-x",
        "ustc-community/dfine-xlarge-coco",
        "ea4f6be7350bbe3c199ec6febc74168346cb5a68",
        "D-FINE X COCO checkpoint",
        "f0f4c9d4364d6ad19d8c7f79d646f77611d61771c5bd06a895fa3d7f64c3e567",
        6873,
        "3ac6ea37c6c995ed65a10beeda703249e97aac1bb40bda18cfaa1c2f6cdca4c5",
        251619400,
        "7e9e2245806aa3ef068a5321f5eebee3116fa4cec560173231954ffaca930eca",
        470
    ),
];

/// The base weights a run of `family` / `size` starts from: the Ultralytics `.pt` export pin
/// (`model:yolo26n`, ...) or a [`TRAIN_BASE_WEIGHTS`] entry.
pub fn base_weights(family: TrainFamily, size: &str) -> Option<&'static Resource> {
    let size = family.size(size)?;
    match family.trainer() {
        Trainer::Ultralytics => export_model(&family.base_name(size)),
        _ => {
            let key = match family {
                TrainFamily::Rfdetr => format!("rfdetr-{size}"),
                _ => format!("{}-{size}", family.as_str()),
            };
            TRAIN_BASE_WEIGHTS
                .iter()
                .find(|r| r.id.strip_prefix(TRAIN_BASE_PREFIX) == Some(key.as_str()))
        }
    }
}

/// A base weights resource by id (`train-base:rfdetr-nano`, `model:yolo26s`), for an explicit
/// `base_weights` in a recipe.
pub fn find_base_weights(id: &str) -> Option<&'static Resource> {
    let id = id.trim();
    TRAIN_BASE_WEIGHTS
        .iter()
        .find(|r| r.id.eq_ignore_ascii_case(id))
        .or_else(|| {
            export_model(id).filter(|r| export_recipe(r) == Some(ExportRecipe::Ultralytics))
        })
}

/// Embedded manifest of the public night training subset (benchmark-set format).
pub const LILA_NIGHT_MANIFEST: &str = include_str!("../../../assets/train/lila-night.json");
/// Id of the night subset resource.
pub const LILA_NIGHT_ID: &str = "train-data:lila-night";
/// Install directory of the night subset under the download root.
pub const LILA_NIGHT_DEST: &str = "train-data/lila-night";

/// The pinned public night subset (LILA BC camera traps, CDLA-Permissive-1.0): plain image files
/// with their boxes in the manifest, installed into `train-data/lila-night/`. Built by
/// `scripts/build_bench_sets.py --set lila-night-train`; over [`LARGE_DOWNLOAD_BYTES`], so
/// [`Resource::is_large`]. None when the embedded manifest lists no image.
pub fn lila_night() -> Option<&'static Resource> {
    static R: std::sync::OnceLock<Option<Resource>> = std::sync::OnceLock::new();
    R.get_or_init(|| {
        let m = lila_night_manifest()?;
        let base = bench_resource(m);
        (!base.parts.is_empty()).then_some(Resource {
            id: LILA_NIGHT_ID,
            dest: LILA_NIGHT_DEST,
            ..base
        })
    })
    .as_ref()
}

/// The parsed night subset manifest (None if it does not parse, which a test rules out).
pub fn lila_night_manifest() -> Option<&'static crate::benchmark::images::Manifest> {
    static M: std::sync::OnceLock<Option<crate::benchmark::images::Manifest>> =
        std::sync::OnceLock::new();
    M.get_or_init(|| crate::benchmark::images::Manifest::parse(LILA_NIGHT_MANIFEST).ok())
        .as_ref()
}

#[cfg(test)]
mod train_tests {
    use super::*;

    #[test]
    fn every_family_and_size_has_pinned_base_weights() {
        for f in TrainFamily::ALL {
            for s in f.sizes() {
                let r = base_weights(f, s).unwrap_or_else(|| panic!("{f} {s}"));
                assert!(!r.parts.is_empty());
                for p in r.parts {
                    assert!(p.url.starts_with("https://"), "{}", p.url);
                    assert_eq!(p.sha256.len(), 64);
                    assert!(p.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
                    assert!(p.size > 0);
                    assert_eq!(p.layout, Layout::File);
                }
                assert!(r.kind.is_plain_files());
                assert_eq!(find_base_weights(r.id).map(|x| x.id), Some(r.id));
            }
        }
        assert_eq!(
            base_weights(TrainFamily::Yolo26, "s").unwrap().id,
            "model:yolo26s"
        );
        assert_eq!(
            base_weights(TrainFamily::Rfdetr, "n").unwrap().id,
            "train-base:rfdetr-nano"
        );
        assert_eq!(
            base_weights(TrainFamily::Dfine, "small").unwrap().id,
            "train-base:dfine-s"
        );
        assert!(base_weights(TrainFamily::Rtdetrv2, "nano").is_none());
        assert!(find_base_weights("model:deimv2-n").is_none());
    }

    #[test]
    fn families_parse_and_spell_sizes() {
        assert_eq!(TrainFamily::parse("RT-DETRv2"), Some(TrainFamily::Rtdetrv2));
        assert_eq!(TrainFamily::parse("d-fine"), Some(TrainFamily::Dfine));
        assert_eq!(TrainFamily::parse("deimv2"), None);
        assert_eq!(TrainFamily::Yolo26.size("small"), Some("s"));
        assert_eq!(TrainFamily::Rfdetr.size("s"), Some("small"));
        assert_eq!(TrainFamily::Rfdetr.size("xl"), None);
        assert_eq!(TrainFamily::Dfine.size("x"), Some("x"));
        assert_eq!(TrainFamily::Dfine.size("n"), None);
        assert_eq!(TrainFamily::Yolo11.license(), AGPL);
        assert_eq!(TrainFamily::Rtdetrv2.license(), APACHE);
        assert_eq!(
            TrainFamily::Rfdetr.service_family(),
            ModelFamilyKind::RfDetr
        );
        assert_eq!(TrainFamily::Dfine.service_family(), ModelFamilyKind::Detr);
        assert_eq!(TrainFamily::Yolov8.service_family(), ModelFamilyKind::Yolo8);
        assert_eq!(TrainFamily::Rfdetr.default_imgsz("base"), 560);
        assert_eq!(TrainFamily::Rfdetr.imgsz_step("base"), 56);
        assert!(!TrainFamily::Dfine.mps_ok());
    }

    #[test]
    fn night_subset_manifest_parses() {
        let m = lila_night_manifest().expect("assets/train/lila-night.json parses");
        assert_eq!(m.id, "lila-night");
        // The format ships before the images: no resource while the list is empty.
        assert_eq!(lila_night().is_some(), !m.images.is_empty());
    }
}
