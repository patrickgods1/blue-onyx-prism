//! CPU inference thread count: `intra_threads` from the config when set, else all physical
//! cores but one on machines with at least 4 (the OS, the HTTP threads and the desktop keep a
//! core), every core on smaller ones. Shared by the server and the benchmark so benchmark
//! numbers mirror production. OpenVINO still picks *which* cores on hybrid CPUs; only the count
//! is capped.

use std::sync::OnceLock;

/// Physical core count from which one core is kept free when `intra_threads` is 0.
pub const RESERVE_FROM_PHYSICAL: usize = 4;

/// Thread count for a configured `intra_threads` on a machine with `physical` cores (pure).
pub fn policy(configured: usize, physical: usize) -> usize {
    if configured > 0 {
        return configured;
    }
    let p = physical.max(1);
    if p >= RESERVE_FROM_PHYSICAL { p - 1 } else { p }
}

/// Physical cores of this machine (at least 1), detected once.
pub fn physical_cores() -> usize {
    static CORES: OnceLock<usize> = OnceLock::new();
    *CORES.get_or_init(|| num_cpus::get_physical().max(1))
}

/// Thread count used by the runtimes for the configured `intra_threads` on this machine.
pub fn effective(configured: usize) -> usize {
    policy(configured, physical_cores())
}

/// Log text for the configured `intra_threads` on this machine, e.g.
/// "5 of 6 physical cores (intra_threads 0 = all but one)".
pub fn describe(configured: usize) -> String {
    describe_for(configured, physical_cores())
}

fn describe_for(configured: usize, physical: usize) -> String {
    let threads = policy(configured, physical);
    let physical = physical.max(1);
    let why = if configured > 0 {
        "intra_threads"
    } else if physical >= RESERVE_FROM_PHYSICAL {
        "intra_threads 0 = all but one"
    } else {
        "intra_threads 0 = all on a machine with fewer than 4"
    };
    format!("{threads} of {physical} physical cores ({why})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_keeps_one_core_from_four() {
        assert_eq!(policy(0, 6), 5);
        assert_eq!(policy(0, 4), 3);
        assert_eq!(policy(0, 3), 3);
        assert_eq!(policy(0, 2), 2);
        assert_eq!(policy(0, 1), 1);
        assert_eq!(policy(0, 0), 1);
    }

    #[test]
    fn policy_honours_configured_value() {
        assert_eq!(policy(3, 16), 3);
        assert_eq!(policy(16, 6), 16);
    }

    #[test]
    fn describe_texts() {
        assert_eq!(
            describe_for(0, 6),
            "5 of 6 physical cores (intra_threads 0 = all but one)"
        );
        assert_eq!(describe_for(4, 6), "4 of 6 physical cores (intra_threads)");
        assert_eq!(
            describe_for(0, 2),
            "2 of 2 physical cores (intra_threads 0 = all on a machine with fewer than 4)"
        );
    }

    #[test]
    fn effective_is_at_least_one() {
        assert!(effective(0) >= 1);
        assert!(physical_cores() >= 1);
    }
}
