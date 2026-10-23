//! Pins of footage import and auto-labelling ([`crate::training::import`],
//! [`crate::training::autolabel`]): the labeler checkpoints ([`AUTOLABEL_WEIGHTS`]: MM-Grounding-DINO,
//! Qwen3-VL, SigLIP 2, each a Hugging Face repo at a commit) and the pinned open footage
//! ([`IMPORT_DATASETS`]: a small MEVA sample). Pure data; downloaded only when the user starts an
//! import or an auto-label run (never by `auto_download`). RF-DETR L, the closed-set labeler, is the
//! training base pin `train-base:rfdetr-large` ([`TRAIN_BASE_WEIGHTS`]).
//!
//! Where the hashes come from: the Hugging Face tree API's LFS `oid`s (SHA-256) of the
//! `*.safetensors` files and the downloaded, hashed small files (configs, tokenizers) at the pinned
//! commits; the MEVA clips were downloaded from the public AWS bucket and hashed.
//!
//! Licences (model cards and dataset page, checked at the pinned commits):
//! - `openmmlab-community/mm_grounding_dino_*`: Apache-2.0 (MM-Grounding-DINO, OpenMMLab).
//! - `Qwen/Qwen3-VL-*-Instruct`: Apache-2.0.
//! - `google/siglip2-base-patch16-224`: Apache-2.0.
//! - MEVA (Kitware / IARPA DIVA, mevadata.org): CC BY 4.0 ("All MEVA data is available for use
//!   under a CC BY-4.0 license").

use super::*;

/// Id prefix of the labeler checkpoints.
pub const AUTOLABEL_PREFIX: &str = "autolabel:";
/// Id prefix of pinned open footage.
pub const IMPORT_DATA_PREFIX: &str = "import-data:";
/// Install directory of the pinned footage under the download root (`<dest>/<key>`).
pub const IMPORT_DATA_DEST: &str = "train-data";

/// A file of a Hugging Face repo at a commit.
macro_rules! hf_part {
    ($repo:literal, $rev:literal, $file:literal, $sha:literal, $size:literal) => {
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

/// A labeler checkpoint: every file of the repo at the commit except `README.md`.
macro_rules! labeler {
    ($key:literal, $repo:literal, $rev:literal, $title:literal, $license:expr, $parts:expr) => {
        Resource {
            id: concat!("autolabel:", $key),
            kind: ResourceKind::ExportModel,
            version: concat!("hf-", $rev),
            platform: None,
            parts: $parts,
            dest: TRAIN_WEIGHTS_DEST,
            provides: Provides::Tool {
                name: concat!("autolabel-", $key),
            },
            title: $title,
            description: concat!($repo, " at ", $rev, " (auto-labelling)"),
            license: $license,
        }
    };
}

/// The labeler checkpoints (see the module docs).
pub static AUTOLABEL_WEIGHTS: &[Resource] = &[
    labeler!(
        "mm-gdino-tiny",
        "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
        "ef2053aa1e0cc9d9cee1df31ca919ee9aae4fc8e",
        "MM-Grounding-DINO Swin-T (O365, GoldG, V3Det)",
        APACHE,
        &[
            hf_part!(
                "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
                "ef2053aa1e0cc9d9cee1df31ca919ee9aae4fc8e",
                "config.json",
                "cb8ffc83fa083aebe4a72d4714c759568b4001b32e28a1850e8c2f30c1db5923",
                2697
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
                "ef2053aa1e0cc9d9cee1df31ca919ee9aae4fc8e",
                "model.safetensors",
                "ad9d9f5f8a94f1bb1ca1969e52e47ce31e97875f80ee07740c74e341839b8755",
                692015412
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
                "ef2053aa1e0cc9d9cee1df31ca919ee9aae4fc8e",
                "preprocessor_config.json",
                "c17fd68afb1f124bfb87a494d409925eec4201b3487c8a694dde64d9ce7109a3",
                511
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
                "ef2053aa1e0cc9d9cee1df31ca919ee9aae4fc8e",
                "special_tokens_map.json",
                "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
                125
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
                "ef2053aa1e0cc9d9cee1df31ca919ee9aae4fc8e",
                "tokenizer_config.json",
                "03070b03f452696da3d89489fa8a4e53730ab74ed2ae466d53923ab5376ce7b5",
                1319
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
                "ef2053aa1e0cc9d9cee1df31ca919ee9aae4fc8e",
                "vocab.txt",
                "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
                231508
            ),
        ]
    ),
    labeler!(
        "mm-gdino-base",
        "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
        "6f52a61753b22842954209e4b190b22135c57bf0",
        "MM-Grounding-DINO Swin-B (O365, GoldG, V3Det)",
        APACHE,
        &[
            hf_part!(
                "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
                "6f52a61753b22842954209e4b190b22135c57bf0",
                "config.json",
                "3e48e01f93f7d94b3b5301e19da35ef6b4a2b6f5c10a3c14d8e39bb72ec92291",
                2701
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
                "6f52a61753b22842954209e4b190b22135c57bf0",
                "model.safetensors",
                "c0b2e1d6ba3ff6c5f2737d22830157ab9859b23db6cb84e513ec3b1da66f5d8d",
                936057092
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
                "6f52a61753b22842954209e4b190b22135c57bf0",
                "preprocessor_config.json",
                "c17fd68afb1f124bfb87a494d409925eec4201b3487c8a694dde64d9ce7109a3",
                511
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
                "6f52a61753b22842954209e4b190b22135c57bf0",
                "special_tokens_map.json",
                "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
                125
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
                "6f52a61753b22842954209e4b190b22135c57bf0",
                "tokenizer_config.json",
                "03070b03f452696da3d89489fa8a4e53730ab74ed2ae466d53923ab5376ce7b5",
                1319
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det",
                "6f52a61753b22842954209e4b190b22135c57bf0",
                "vocab.txt",
                "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
                231508
            ),
        ]
    ),
    labeler!(
        "mm-gdino-large",
        "openmmlab-community/mm_grounding_dino_large_all",
        "0445d0a0ae5f0fa6be689cf7243d4d0ef4ba7de0",
        "MM-Grounding-DINO Swin-L (all data)",
        APACHE,
        &[
            hf_part!(
                "openmmlab-community/mm_grounding_dino_large_all",
                "0445d0a0ae5f0fa6be689cf7243d4d0ef4ba7de0",
                "config.json",
                "7669f371f9ae2d642881ecd7b1769095384439c8a9e01b5f958eaed298919cb2",
                2727
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_large_all",
                "0445d0a0ae5f0fa6be689cf7243d4d0ef4ba7de0",
                "model.safetensors",
                "1802856b98c66cdf6c4b1e9841320f7b76b039c9cb54e88061de1c2517b53822",
                1376363740
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_large_all",
                "0445d0a0ae5f0fa6be689cf7243d4d0ef4ba7de0",
                "preprocessor_config.json",
                "c17fd68afb1f124bfb87a494d409925eec4201b3487c8a694dde64d9ce7109a3",
                511
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_large_all",
                "0445d0a0ae5f0fa6be689cf7243d4d0ef4ba7de0",
                "special_tokens_map.json",
                "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
                125
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_large_all",
                "0445d0a0ae5f0fa6be689cf7243d4d0ef4ba7de0",
                "tokenizer_config.json",
                "03070b03f452696da3d89489fa8a4e53730ab74ed2ae466d53923ab5376ce7b5",
                1319
            ),
            hf_part!(
                "openmmlab-community/mm_grounding_dino_large_all",
                "0445d0a0ae5f0fa6be689cf7243d4d0ef4ba7de0",
                "vocab.txt",
                "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
                231508
            ),
        ]
    ),
    labeler!(
        "qwen3-vl-2b",
        "Qwen/Qwen3-VL-2B-Instruct",
        "89644892e4d85e24eaac8bacfd4f463576704203",
        "Qwen3-VL 2B Instruct",
        APACHE,
        &[
            hf_part!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "chat_template.json",
                "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
                5502
            ),
            hf_part!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "config.json",
                "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
                1505
            ),
            hf_part!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "generation_config.json",
                "1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4",
                269
            ),
            hf_part!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "merges.txt",
                "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
                1671839
            ),
            hf_part!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "model.safetensors",
                "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
                4255140312
            ),
            hf_part!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "preprocessor_config.json",
                "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
                390
            ),
            hf_part!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "tokenizer.json",
                "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
                7032403
            ),
            hf_part!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "tokenizer_config.json",
                "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
                10868
            ),
            hf_part!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "video_preprocessor_config.json",
                "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
                385
            ),
            hf_part!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "vocab.json",
                "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
                2776833
            ),
        ]
    ),
    labeler!(
        "qwen3-vl-4b",
        "Qwen/Qwen3-VL-4B-Instruct",
        "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "Qwen3-VL 4B Instruct",
        APACHE,
        &[
            hf_part!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "chat_template.json",
                "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
                5502
            ),
            hf_part!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "config.json",
                "edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d",
                1505
            ),
            hf_part!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "generation_config.json",
                "8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44",
                269
            ),
            hf_part!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "merges.txt",
                "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
                1671839
            ),
            hf_part!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "model-00001-of-00002.safetensors",
                "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
                4967229296
            ),
            hf_part!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "model-00002-of-00002.safetensors",
                "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
                3908490048
            ),
            hf_part!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "model.safetensors.index.json",
                "58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b",
                64742
            ),
            hf_part!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "preprocessor_config.json",
                "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
                390
            ),
            hf_part!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "tokenizer.json",
                "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
                7032403
            ),
            hf_part!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "tokenizer_config.json",
                "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
                10868
            ),
            hf_part!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "video_preprocessor_config.json",
                "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
                385
            ),
            hf_part!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "vocab.json",
                "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
                2776833
            ),
        ]
    ),
    labeler!(
        "qwen3-vl-8b",
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "Qwen3-VL 8B Instruct",
        APACHE,
        &[
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "chat_template.json",
                "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
                5499
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "config.json",
                "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661",
                1474
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "generation_config.json",
                "8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44",
                269
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "merges.txt",
                "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
                1671839
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "model-00001-of-00004.safetensors",
                "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa",
                4902275944
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "model-00002-of-00004.safetensors",
                "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5",
                4915962496
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "model-00003-of-00004.safetensors",
                "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192",
                4999831048
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "model-00004-of-00004.safetensors",
                "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5",
                2716270024
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "model.safetensors.index.json",
                "520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070",
                67759
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "preprocessor_config.json",
                "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
                390
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "tokenizer.json",
                "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
                7032403
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "tokenizer_config.json",
                "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
                10868
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "video_preprocessor_config.json",
                "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
                385
            ),
            hf_part!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "vocab.json",
                "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
                2776833
            ),
        ]
    ),
    labeler!(
        "siglip2-base",
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "SigLIP 2 base (patch 16, 224 px)",
        APACHE,
        &[
            hf_part!(
                "google/siglip2-base-patch16-224",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "config.json",
                "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7",
                253
            ),
            hf_part!(
                "google/siglip2-base-patch16-224",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "model.safetensors",
                "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
                1500800904
            ),
            hf_part!(
                "google/siglip2-base-patch16-224",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "preprocessor_config.json",
                "9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc",
                394
            ),
            hf_part!(
                "google/siglip2-base-patch16-224",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "special_tokens_map.json",
                "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
                636
            ),
            hf_part!(
                "google/siglip2-base-patch16-224",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "tokenizer.json",
                "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
                34363039
            ),
            hf_part!(
                "google/siglip2-base-patch16-224",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "tokenizer.model",
                "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
                4241003
            ),
            hf_part!(
                "google/siglip2-base-patch16-224",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "tokenizer_config.json",
                "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8",
                47164
            ),
        ]
    ),
];

/// A labeler checkpoint by key (`mm-gdino-tiny`, `qwen3-vl-2b`, `siglip2-base`) or id.
pub fn autolabel_weights(key: &str) -> Option<&'static Resource> {
    let k = key.trim().to_ascii_lowercase();
    AUTOLABEL_WEIGHTS
        .iter()
        .find(|r| r.id == k || r.id.strip_prefix(AUTOLABEL_PREFIX) == Some(k.as_str()))
}

/// Licence of the pinned MEVA sample.
pub const MEVA_LICENSE: &str = "CC-BY-4.0";
/// Attribution of the MEVA sample (CC BY 4.0 asks for it).
pub const MEVA_ATTRIBUTION: &str = "MEVA (Multiview Extended Video with Activities), Kitware Inc. and     IARPA DIVA, https://mevadata.org, CC BY 4.0";

/// Pinned open footage: six five-minute MEVA clips (CC BY 4.0, fixed outdoor surveillance cameras at
/// 1920x1072 or 352x240): a night IR camera (G639, monochrome), two thermal cameras (G475, G476), a
/// lit night camera (G423), and two day cameras (G419, G508). About 307 MB.
pub static IMPORT_DATASETS: &[Resource] = &[Resource {
    id: "import-data:meva-sample",
    kind: ResourceKind::ExportModel,
    version: "drops-123-r13",
    platform: None,
    parts: &[
        Part {
            url: "https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-11/00/2018-03-11.23-55-00.00-00-00.school.G639.r13.avi",
            sha256: "4486246ff4d5856d59ea6fb4ba3541e69d7202600eebfa9ff2d0ad484fbcb808",
            size: 53111976,
            file_name: "2018-03-11.23-55-00.00-00-00.school.G639.r13.avi",
            archive: None,
            layout: Layout::File,
        },
        Part {
            url: "https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-11/00/2018-03-11.23-55-01.00-00-01.school.G423.r13.avi",
            sha256: "5855f663e3411984128fa7950c57d13837d07fe65893f3930c5ace01d6eea8cf",
            size: 78385940,
            file_name: "2018-03-11.23-55-01.00-00-01.school.G423.r13.avi",
            archive: None,
            layout: Layout::File,
        },
        Part {
            url: "https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-11/00/2018-03-11.23-55-00.00-00-00.bus.G475.r13.avi",
            sha256: "efb1478410a7b3192312677c2fb6d3cbd21f36f03d19cb366f78db6bb3cf0d34",
            size: 1513068,
            file_name: "2018-03-11.23-55-00.00-00-00.bus.G475.r13.avi",
            archive: None,
            layout: Layout::File,
        },
        Part {
            url: "https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-11/00/2018-03-11.23-55-00.00-00-00.hospital.G476.r13.avi",
            sha256: "c7efd83e4cca3d899e19cf36748035e1633915255e03a4320b5535fae717cccf",
            size: 1834712,
            file_name: "2018-03-11.23-55-00.00-00-00.hospital.G476.r13.avi",
            archive: None,
            layout: Layout::File,
        },
        Part {
            url: "https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-07/17/2018-03-07.17-00-00.17-05-00.school.G419.r13.avi",
            sha256: "4ba5cc0d2acb9b219105281168c4520f477ebb5d4ca36aa8126f190d3ddb6b63",
            size: 69851730,
            file_name: "2018-03-07.17-00-00.17-05-00.school.G419.r13.avi",
            archive: None,
            layout: Layout::File,
        },
        Part {
            url: "https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-07/17/2018-03-07.17-00-00.17-05-00.bus.G508.r13.avi",
            sha256: "e4ef161838c865d8d92552bf7589f5ccfeb9d45245449da317bb5b4be224b7aa",
            size: 102847934,
            file_name: "2018-03-07.17-00-00.17-05-00.bus.G508.r13.avi",
            archive: None,
            layout: Layout::File,
        },
    ],
    dest: IMPORT_DATA_DEST,
    provides: Provides::Tool {
        name: "import-meva-sample",
    },
    title: "MEVA sample (6 clips, night IR, thermal and day)",
    description: "Six 5-minute clips of the MEVA dataset's fixed cameras (mevadata.org), CC BY 4.0",
    license: MEVA_LICENSE,
}];

/// Pinned footage by key (`meva-sample`) or id.
pub fn import_dataset(key: &str) -> Option<&'static Resource> {
    let k = key.trim().to_ascii_lowercase();
    IMPORT_DATASETS
        .iter()
        .find(|r| r.id == k || r.id.strip_prefix(IMPORT_DATA_PREFIX) == Some(k.as_str()))
}

/// Install directory key of a labeler or footage resource (`mm-gdino-tiny`, `meva-sample`): the id
/// without its prefix, safe as a folder name on every OS.
pub fn install_key(res: &Resource) -> &'static str {
    res.id
        .strip_prefix(AUTOLABEL_PREFIX)
        .or_else(|| res.id.strip_prefix(IMPORT_DATA_PREFIX))
        .unwrap_or(res.id)
}

#[cfg(test)]
mod autolabel_tests {
    use super::*;

    #[test]
    fn labeler_and_footage_pins_are_complete() {
        let mut ids = std::collections::HashSet::new();
        for r in AUTOLABEL_WEIGHTS.iter().chain(IMPORT_DATASETS) {
            assert!(ids.insert(r.id), "duplicate {}", r.id);
            assert!(r.kind.is_plain_files(), "{}", r.id);
            assert!(!r.parts.is_empty(), "{}", r.id);
            let mut names = std::collections::HashSet::new();
            for p in r.parts {
                assert!(p.url.starts_with("https://"), "{}", p.url);
                assert_eq!(p.sha256.len(), 64, "{}", p.url);
                assert!(p.sha256.bytes().all(|b| b.is_ascii_hexdigit()), "{}", p.url);
                assert!(p.size > 0, "{}", p.url);
                assert!(names.insert(p.file_name), "{}: {} twice", r.id, p.file_name);
                assert!(p.url.ends_with(p.file_name), "{}", p.url);
                assert!(!p.file_name.contains('/') && !p.file_name.contains('\\'));
            }
            assert!(!install_key(r).contains(':'), "{}", r.id);
        }
        for r in AUTOLABEL_WEIGHTS {
            assert_eq!(r.license, APACHE, "{}", r.id);
            assert!(
                r.parts.iter().any(|p| p.file_name == "config.json"),
                "{}",
                r.id
            );
            assert!(
                r.parts
                    .iter()
                    .any(|p| p.file_name.ends_with(".safetensors")),
                "{}",
                r.id
            );
            // Pinned to a commit, never a branch.
            assert!(r.parts.iter().all(|p| !p.url.contains("/resolve/main/")));
        }
        assert_eq!(
            autolabel_weights("MM-GDINO-TINY").map(|r| r.id),
            Some("autolabel:mm-gdino-tiny")
        );
        assert_eq!(
            autolabel_weights("autolabel:qwen3-vl-2b").map(install_key),
            Some("qwen3-vl-2b")
        );
        assert!(autolabel_weights("gdino").is_none());
        let meva = import_dataset("meva-sample").unwrap();
        assert_eq!(meva.license, MEVA_LICENSE);
        assert_eq!(meva.parts.len(), 6);
        assert!(meva.size() < 1_000_000_000);
        // The four the README lists as large downloads are over the gate.
        assert!(autolabel_weights("qwen3-vl-2b").unwrap().is_large());
    }
}
