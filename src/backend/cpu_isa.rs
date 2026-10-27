//! oneDNN CPU ISA cap for OpenVINO's CPU plugin: on Windows x86_64, a CPU with AMX-TILE but no
//! AMX-FP16 (Sapphire / Emerald Rapids class, e.g. Xeon Platinum 8573C) makes OpenVINO's AMX
//! (brgemm) JIT kernels fault with STATUS_ILLEGAL_INSTRUCTION at `tilezero` on the first bf16
//! inference. Before OpenVINO loads we set `ONEDNN_MAX_CPU_ISA=AVX512_CORE_FP16` (AMX off,
//! AVX-512 FP16/BF16 kept) unless the user already set it. The decision ([`decide`]) is pure and
//! tested on every OS; only the CPUID read and the `set_var` are platform-gated.

use std::sync::OnceLock;

/// Environment variable oneDNN (inside OpenVINO's CPU plugin) reads to cap the JIT ISA.
pub const ENV_MAX_CPU_ISA: &str = "ONEDNN_MAX_CPU_ISA";
/// The cap applied on affected CPUs: highest ISA below AMX, keeps AVX-512 FP16/BF16.
pub const CAPPED_ISA: &str = "AVX512_CORE_FP16";

/// CPUID facts that matter for the decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmxFacts {
    /// CPUID.(7,0):EDX bit 24.
    pub amx_tile: bool,
    /// CPUID.(7,1):EAX bit 21.
    pub amx_fp16: bool,
}

/// What [`decide`] chose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IsaAction {
    /// Nothing to do (not affected).
    None,
    /// Set `ONEDNN_MAX_CPU_ISA` to this value.
    Cap(&'static str),
    /// Affected CPU, but the user already set `ONEDNN_MAX_CPU_ISA` to this value.
    UserSet(String),
}

/// Pure decision: `facts` is `None` when CPUID is unavailable (not x86_64 or leaf 7 missing),
/// `windows` whether this is Windows, `existing` the current `ONEDNN_MAX_CPU_ISA` (an empty
/// value counts as unset).
pub fn decide(facts: Option<AmxFacts>, windows: bool, existing: Option<&str>) -> IsaAction {
    let affected = windows && facts.is_some_and(|f| f.amx_tile && !f.amx_fp16);
    if !affected {
        return IsaAction::None;
    }
    match existing.filter(|v| !v.is_empty()) {
        Some(v) => IsaAction::UserSet(v.to_string()),
        None => IsaAction::Cap(CAPPED_ISA),
    }
}

/// Reads the AMX bits with CPUID; `None` when leaf 7 is not supported.
#[cfg(all(windows, target_arch = "x86_64"))]
fn read_facts() -> Option<AmxFacts> {
    use std::arch::x86_64::{__cpuid, __cpuid_count};
    let max_leaf = __cpuid(0).eax;
    if max_leaf < 7 {
        return None;
    }
    let l7 = __cpuid_count(7, 0);
    let amx_tile = l7.edx & (1 << 24) != 0;
    // Sub-leaf 1 exists when CPUID.(7,0):EAX (max sub-leaf) >= 1.
    let amx_fp16 = l7.eax >= 1 && __cpuid_count(7, 1).eax & (1 << 21) != 0;
    Some(AmxFacts { amx_tile, amx_fp16 })
}

#[cfg(not(all(windows, target_arch = "x86_64")))]
fn read_facts() -> Option<AmxFacts> {
    None
}

/// Applies the decision once per process and returns a note for `libs::diagnostics()`
/// (`None` when the CPU is not affected). Later calls return the same note without re-deciding,
/// so a value we set ourselves is never reported as user-set.
pub(crate) fn apply() -> Option<String> {
    static NOTE: OnceLock<Option<String>> = OnceLock::new();
    NOTE.get_or_init(|| {
        let existing = std::env::var(ENV_MAX_CPU_ISA).ok();
        match decide(read_facts(), cfg!(windows), existing.as_deref()) {
            IsaAction::None => None,
            IsaAction::Cap(isa) => {
                set_isa(isa);
                let msg = format!(
                    "set {ENV_MAX_CPU_ISA}={isa}: this CPU has AMX but no AMX-FP16, and OpenVINO's AMX \
                     kernels fault with an illegal instruction on Windows on this CPU class; set \
                     {ENV_MAX_CPU_ISA} yourself (e.g. AVX512_CORE_AMX) to override"
                );
                tracing::info!("{msg}");
                Some(msg)
            }
            IsaAction::UserSet(v) => {
                let msg = format!(
                    "{ENV_MAX_CPU_ISA} already set to {v} (left untouched; this CPU has AMX but no AMX-FP16)"
                );
                tracing::debug!("{msg}");
                Some(msg)
            }
        }
    })
    .clone()
}

#[cfg(windows)]
fn set_isa(isa: &str) {
    // SAFETY: the Windows environment block is protected by a lock (SetEnvironmentVariableW),
    // so this is sound even with other threads running.
    unsafe { std::env::set_var(ENV_MAX_CPU_ISA, isa) };
}

#[cfg(not(windows))]
fn set_isa(_isa: &str) {
    // `decide` never caps off Windows.
}

#[cfg(test)]
mod tests {
    use super::*;

    const AMX_NO_FP16: AmxFacts = AmxFacts {
        amx_tile: true,
        amx_fp16: false,
    };
    const AMX_FP16: AmxFacts = AmxFacts {
        amx_tile: true,
        amx_fp16: true,
    };
    const NO_AMX: AmxFacts = AmxFacts {
        amx_tile: false,
        amx_fp16: false,
    };

    #[test]
    fn amx_without_fp16_on_windows_is_capped() {
        assert_eq!(
            decide(Some(AMX_NO_FP16), true, None),
            IsaAction::Cap("AVX512_CORE_FP16")
        );
    }

    #[test]
    fn amx_with_fp16_is_left_alone() {
        assert_eq!(decide(Some(AMX_FP16), true, None), IsaAction::None);
    }

    #[test]
    fn no_amx_or_no_cpuid_is_left_alone() {
        assert_eq!(decide(Some(NO_AMX), true, None), IsaAction::None);
        assert_eq!(decide(None, true, None), IsaAction::None);
    }

    #[test]
    fn other_os_is_left_alone() {
        assert_eq!(decide(Some(AMX_NO_FP16), false, None), IsaAction::None);
    }

    #[test]
    fn user_value_is_untouched() {
        assert_eq!(
            decide(Some(AMX_NO_FP16), true, Some("AVX512_CORE_AMX")),
            IsaAction::UserSet("AVX512_CORE_AMX".into())
        );
    }

    #[test]
    fn empty_user_value_counts_as_unset() {
        assert_eq!(
            decide(Some(AMX_NO_FP16), true, Some("")),
            IsaAction::Cap(CAPPED_ISA)
        );
    }

    #[test]
    fn apply_off_windows_x86_does_nothing() {
        if !cfg!(all(windows, target_arch = "x86_64")) {
            assert_eq!(apply(), None);
        }
    }
}
