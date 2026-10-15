//! Runtime pins: the OpenVINO runtime archives ([`OPENVINO`]), the ONNX Runtime and DirectML
//! packages ([`ONNXRUNTIME`]) and NVIDIA's CUDA / cuDNN wheels ([`CUDA_LIBS`]).
//! Pure data: every URL, SHA-256 and size is a reviewed pin; change them only in reviewed commits.

use super::*;

// ---------------------------------------------------------------------------------------------
// OpenVINO runtime
// ---------------------------------------------------------------------------------------------

macro_rules! ov_url {
    ($dir:literal, $file:literal) => {
        concat!(
            "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.4/",
            $dir,
            "/",
            $file
        )
    };
}

macro_rules! ov_part {
    ($dir:literal, $file:literal, $kind:expr, $sha:literal, $size:literal) => {
        Part {
            url: ov_url!($dir, $file),
            sha256: $sha,
            size: $size,
            file_name: $file,
            archive: Some($kind),
            layout: Layout::OpenVino,
        }
    };
}

const OV_GPU_CPU: &[Target] = &[Target::Gpu, Target::Cpu];
const OV_CPU: &[Target] = &[Target::Cpu];

const fn openvino(
    platform: Platform,
    part: &'static [Part],
    targets: &'static [Target],
) -> Resource {
    Resource {
        id: OPENVINO_RUNTIME_ID,
        kind: ResourceKind::OpenVinoRuntime,
        version: OPENVINO_VERSION,
        platform: Some(platform),
        parts: part,
        dest: "openvino",
        provides: Provides::Runtime {
            runtime: Runtime::OpenVino,
            targets,
        },
        title: "OpenVINO runtime",
        description: "Intel OpenVINO runtime libraries (openvino:* devices)",
        license: "Apache-2.0",
    }
}

/// `openvino-runtime` per platform. The Intel GPU plugin ships only for x86_64.
pub(crate) static OPENVINO: &[Resource] = &[
    openvino(
        Platform::WINDOWS_X64,
        &[ov_part!(
            "windows_vc_mt",
            "openvino_toolkit_windows_vc_mt_2026.4.0.22959.99c81491cc3_x86_64.zip",
            ArchiveKind::Zip,
            "82946205ac9f9f8113acf5f7fb5c705ef06a666aa15a8acf52cf2b4e1be90063",
            235127912
        )],
        OV_GPU_CPU,
    ),
    // The RHEL 8 build: it needs glibc 2.27 and GLIBCXX_3.4.22 at most, so it loads on every
    // distribution the deb/rpm support (Ubuntu 22.04+, Debian 12, RHEL 9+). The ubuntu24 build
    // needs glibc 2.38 and fails to load there (the service then exits at every start).
    openvino(
        Platform::LINUX_X64,
        &[ov_part!(
            "linux",
            "openvino_toolkit_rhel8_2026.4.0.22959.99c81491cc3_x86_64.tgz",
            ArchiveKind::TarGz,
            "70e40134514d54b291b557ad26f85cc0ad554587ba49686285fee1e0cee8ab79",
            80635452
        )],
        OV_GPU_CPU,
    ),
    openvino(
        Platform::LINUX_ARM64,
        &[ov_part!(
            "linux",
            "openvino_toolkit_ubuntu22_2026.4.0.22959.99c81491cc3_arm64.tgz",
            ArchiveKind::TarGz,
            "ed3f103ae8dd7a509c23645f0d34983378d3e609711a82d65301ac77d4a34800",
            43359520
        )],
        OV_CPU,
    ),
    openvino(
        Platform::MACOS_ARM64,
        &[ov_part!(
            "macos",
            "openvino_toolkit_macos_12_6_2026.4.0.22959.99c81491cc3_arm64.tgz",
            ArchiveKind::TarGz,
            "1e67db1fa25e8d736f35f5b3d663d2fd669d51c0d2892f682d0facc9c22b019c",
            44413951
        )],
        OV_CPU,
    ),
];

// ---------------------------------------------------------------------------------------------
// ONNX Runtime
// ---------------------------------------------------------------------------------------------

macro_rules! ort_release {
    ($ver:literal, $file:literal, $kind:expr, $sha:literal, $size:literal) => {
        Part {
            url: concat!(
                "https://github.com/microsoft/onnxruntime/releases/download/v",
                $ver,
                "/",
                $file
            ),
            sha256: $sha,
            size: $size,
            file_name: $file,
            archive: Some($kind),
            layout: Layout::OrtRelease,
        }
    };
}

macro_rules! nuget {
    ($id:literal, $ver:literal, $layout:expr, $sha:literal, $size:literal) => {
        Part {
            url: concat!(
                "https://api.nuget.org/v3-flatcontainer/",
                $id,
                "/",
                $ver,
                "/",
                $id,
                ".",
                $ver,
                ".nupkg"
            ),
            sha256: $sha,
            size: $size,
            file_name: concat!($id, ".", $ver, ".nupkg"),
            archive: Some(ArchiveKind::Zip),
            layout: $layout,
        }
    };
}

const fn ort(
    platform: Platform,
    flavor: Flavor,
    parts: &'static [Part],
    title: &'static str,
) -> Resource {
    Resource {
        id: flavor.resource_id(),
        kind: ResourceKind::OnnxRuntime(flavor),
        version: onnxruntime_version(flavor),
        platform: Some(platform),
        parts,
        dest: flavor.dest(),
        provides: Provides::Runtime {
            runtime: Runtime::Ort,
            targets: flavor.targets(),
        },
        title,
        description: "ONNX Runtime shared libraries (ort:* devices); one flavor is active per process",
        license: "MIT",
    }
}

const ORT_OSX_ARM64: Part = ort_release!(
    "1.24.4",
    "onnxruntime-osx-arm64-1.24.4.tgz",
    ArchiveKind::TarGz,
    "93787795f47e1eee369182e43ed51b9e5da0878ab0346aecf4258979b8bba989",
    30937282
);

/// `onnxruntime-<flavor>` per platform (the combinations `setup-onnxruntime` supports).
pub(crate) static ONNXRUNTIME: &[Resource] = &[
    ort(
        Platform::WINDOWS_X64,
        Flavor::Cpu,
        &[ort_release!(
            "1.24.4",
            "onnxruntime-win-x64-1.24.4.zip",
            ArchiveKind::Zip,
            "d2319fddfb6ea4db99ccc4b60c85c517bcd855721f5daa6a06d40d7cb2ee2357",
            74442783
        )],
        "ONNX Runtime CPU",
    ),
    ort(
        Platform::WINDOWS_X64,
        Flavor::Cuda,
        &[ort_release!(
            "1.31.0",
            "onnxruntime-win-x64-gpu_cuda12-1.31.0.zip",
            ArchiveKind::Zip,
            "fa79097fa478961cbd31c297671ac169eb0d3f1dfdc9e8481e091850f8c4810b",
            392994831
        )],
        "ONNX Runtime CUDA/TensorRT (CUDA 12)",
    ),
    ort(
        Platform::WINDOWS_X64,
        Flavor::DirectMl,
        &[
            nuget!(
                "microsoft.ml.onnxruntime.directml",
                "1.24.4",
                Layout::NugetOrt,
                "57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681",
                12458649
            ),
            nuget!(
                "microsoft.ai.directml",
                "1.15.4",
                Layout::NugetDirectMl,
                "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9",
                202292617
            ),
        ],
        "ONNX Runtime DirectML",
    ),
    ort(
        Platform::LINUX_X64,
        Flavor::Cpu,
        &[ort_release!(
            "1.24.4",
            "onnxruntime-linux-x64-1.24.4.tgz",
            ArchiveKind::TarGz,
            "3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747",
            8155822
        )],
        "ONNX Runtime CPU",
    ),
    ort(
        Platform::LINUX_X64,
        Flavor::Cuda,
        &[ort_release!(
            "1.31.0",
            "onnxruntime-linux-x64-gpu_cuda12-1.31.0.tgz",
            ArchiveKind::TarGz,
            "62d3af5269af2143b5d37494337573f1ffb8885f3ad7477ae0cb1cc79c8dedfd",
            450228883
        )],
        "ONNX Runtime CUDA/TensorRT (CUDA 12)",
    ),
    ort(
        Platform::LINUX_ARM64,
        Flavor::Cpu,
        &[ort_release!(
            "1.24.4",
            "onnxruntime-linux-aarch64-1.24.4.tgz",
            ArchiveKind::TarGz,
            "866109a9248d057671a039b9d725be4bd86888e3754140e6701ec621be9d4d7e",
            7181958
        )],
        "ONNX Runtime CPU",
    ),
    ort(
        Platform::MACOS_ARM64,
        Flavor::CoreMl,
        &[ORT_OSX_ARM64],
        "ONNX Runtime CoreML",
    ),
    // The macOS package has both EPs; `cpu` is the same archive.
    ort(
        Platform::MACOS_ARM64,
        Flavor::Cpu,
        &[ORT_OSX_ARM64],
        "ONNX Runtime CPU",
    ),
];

// ---------------------------------------------------------------------------------------------
// NVIDIA CUDA / cuDNN libraries (PyPI wheels)
// ---------------------------------------------------------------------------------------------

macro_rules! wheel {
    ($url:literal, $sha:literal, $size:literal) => {
        Part {
            url: $url,
            sha256: $sha,
            size: $size,
            file_name: wheel_file_name($url),
            archive: Some(ArchiveKind::Zip),
            layout: Layout::NvidiaWheel,
        }
    };
}

/// Last path segment of a URL (const, for the wheel table).
const fn wheel_file_name(url: &'static str) -> &'static str {
    let bytes = url.as_bytes();
    let mut head = bytes;
    while let [rest @ .., last] = head {
        if *last == b'/' {
            break;
        }
        head = rest;
    }
    let (_, tail) = bytes.split_at(head.len());
    match std::str::from_utf8(tail) {
        Ok(s) => s,
        // Evaluated at compile time for the static wheel table, so this can only ever be a
        // build error, never a runtime panic.
        #[allow(
            clippy::panic,
            reason = "invariant: tail starts after an ASCII '/' (or at 0) of a &str, so it is \
                      always valid UTF-8; const fn cannot return an error"
        )]
        Err(_) => panic!("URL is not UTF-8"),
    }
}

const fn cuda_libs(platform: Platform, parts: &'static [Part]) -> Resource {
    Resource {
        id: CUDA_LIBS_ID,
        kind: ResourceKind::CudaLibs,
        version: CUDA_LIBS_VERSION,
        platform: Some(platform),
        parts,
        dest: CUDA_LIBS_DEST,
        provides: Provides::CudaLibs,
        title: "NVIDIA CUDA 12 + cuDNN 9.10 libraries",
        description: "CUDA runtime, cuBLAS, cuDNN, cuFFT, cuRAND for ort:cuda (large, opt-in)",
        license: "NVIDIA Software License Agreement (CUDA, cuDNN redistributables)",
    }
}

/// `nvidia-cuda-libs`: what ONNX Runtime's CUDA provider uses: cudart, cuBLAS/cuBLASLt, cuRAND
/// and NVRTC are linked (import tables of the 1.31.0 provider libraries), cuDNN 9 and cuFFT are
/// loaded at run time (linked in 1.24.4), plus nvJitLink (cuFFT's dependency). cuDNN stays at
/// 9.10 (`CUDNN_VERSION`): the last with Pascal kernels. The driver's `libcuda` / `nvcuda.dll` is
/// not included.
pub(crate) static CUDA_LIBS: &[Resource] = &[
    cuda_libs(
        Platform::WINDOWS_X64,
        &[
            wheel!(
                "https://files.pythonhosted.org/packages/30/a5/a515b7600ad361ea14bfa13fb4d6687abf500adc270f19e89849c0590492/nvidia_cuda_runtime_cu12-12.8.90-py3-none-win_amd64.whl",
                "c0c6027f01505bfed6c3b21ec546f69c687689aad5f1a377554bc6ca4aa993a8",
                944318
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/70/61/7d7b3c70186fb651d0fbd35b01dbfc8e755f69fd58f817f3d0f642df20c3/nvidia_cublas_cu12-12.8.4.1-py3-none-win_amd64.whl",
                "47e9b82132fa8d2b4944e708049229601448aaad7e6f296f630f2d1a32de35af",
                567544208
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/3d/90/0bd6e586701b3a890fd38aa71c387dab4883d619d6e5ad912ccbd05bfd67/nvidia_cudnn_cu12-9.10.2.21-py3-none-win_amd64.whl",
                "c6288de7d63e6cf62988f0923f96dc339cea362decb1bf5b3141883392a7d65e",
                692992268
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/7d/ec/ce1629f1e478bb5ccd208986b5f9e0316a78538dd6ab1d0484f012f8e2a1/nvidia_cufft_cu12-11.3.3.83-py3-none-win_amd64.whl",
                "7a64a98ef2a7c47f905aaf8931b69a3a43f27c55530c698bb2ed7c75c0b42cb7",
                192216559
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/b9/75/70c05b2f3ed5be3bb30b7102b6eb78e100da4bbf6944fd6725c012831cab/nvidia_curand_cu12-10.3.9.90-py3-none-win_amd64.whl",
                "f149a8ca457277da854f89cf282d6ef43176861926c7ac85b2a0fbd237c587ec",
                62765309
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/ed/d7/34f02dad2e30c31b10a51f6b04e025e5dd60e5f936af9045a9b858a05383/nvidia_nvjitlink_cu12-12.8.93-py3-none-win_amd64.whl",
                "bd93fbeeee850917903583587f4fc3a4eafa022e34572251368238ab5e6bd67f",
                268553710
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/45/51/52a3d84baa2136cc8df15500ad731d74d3a1114d4c123e043cb608d4a32b/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-win_amd64.whl",
                "7a4b6b2904850fe78e0bd179c4b655c404d4bb799ef03ddc60804247099ae909",
                73586838
            ),
        ],
    ),
    cuda_libs(
        Platform::LINUX_X64,
        &[
            wheel!(
                "https://files.pythonhosted.org/packages/0d/9b/a997b638fcd068ad6e4d53b8551a7d30fe8b404d6f1804abf1df69838932/nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
                "adade8dcbd0edf427b7204d480d6066d33902cab2a4707dcfc48a2d0fd44ab90",
                954765
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/dc/61/e24b560ab2e2eaeb3c839129175fb330dfcfc29e5203196e5541a4c44682/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl",
                "8ac4e771d5a348c551b2a426eda6193c19aa630236b418086020df5ba9667142",
                594346921
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/ba/51/e123d997aa098c61d029f76663dedbfb9bc8dcf8c60cbd6adbe42f76d049/nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl",
                "949452be657fa16687d0930933f032835951ef0892b37d2d53824d1a84dc97a8",
                706758467
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/1f/13/ee4e00f30e676b66ae65b4f08cb5bcbb8392c03f54f2d5413ea99a5d1c80/nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
                "4d2dd21ec0b88cf61b62e6b43564355e5222e4a3fb394cac0db101f2dd0d4f74",
                193118695
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/fb/aa/6584b56dc84ebe9cf93226a5cde4d99080c8e90ab40f0c27bda7a0f29aa1/nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl",
                "b32331d4f4df5d6eefa0554c565b626c7216f87a06a4f56fab27c3b68a830ec9",
                63619976
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/f6/74/86a07f1d0f42998ca31312f998bd3b9a7eff7f52378f4f270c8679c77fb9/nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
                "81ff63371a7ebd6e6451970684f916be2eab07321b73c9d244dc2b4da7f73b88",
                39254836
            ),
            wheel!(
                "https://files.pythonhosted.org/packages/05/6b/32f747947df2da6994e999492ab306a903659555dddc0fbdeb9d71f75e52/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
                "a7756528852ef889772a84c6cd89d41dfa74667e24cca16bb31f8f061e3e9994",
                88040029
            ),
        ],
    ),
];
