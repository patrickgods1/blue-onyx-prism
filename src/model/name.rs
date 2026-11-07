//! Model name normalization, std only so `config.rs` and the registry share it without one
//! depending on the other.

/// Lowercase and strip a trailing `.onnx` / `.xml` (Blue Iris sends file stems, users may not).
pub fn normalize_name(name: &str) -> String {
    let lower = name.trim().to_ascii_lowercase();
    for ext in [".onnx", ".xml"] {
        if let Some(stripped) = lower.strip_suffix(ext)
            && !stripped.is_empty()
        {
            return stripped.to_string();
        }
    }
    lower
}

#[cfg(test)]
mod tests {
    use super::normalize_name;

    #[test]
    fn lowercases_trims_and_strips_one_model_extension() {
        assert_eq!(normalize_name(" IPcam-General.ONNX "), "ipcam-general");
        assert_eq!(normalize_name("a.xml"), "a");
        assert_eq!(normalize_name("a.onnx.xml"), "a.onnx");
        assert_eq!(normalize_name(".onnx"), ".onnx");
        assert_eq!(normalize_name("plain"), "plain");
    }
}
