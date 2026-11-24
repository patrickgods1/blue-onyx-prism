//! Telling a helper's out-of-memory failure from its log lines: the host (RAM + paging file)
//! or the GPU. The one rule the training runs ([`super::job`]: batch halving, worker halving,
//! the failure message) and the import / auto-label / overlap jobs ([`super::pyenv`]: their
//! "use a smaller labeler" advice) share (QA-102: two copies had drifted).

/// A line of `lines` that reports the host out of memory (RAM + paging file, the Windows
/// commit), if any: of the most telling kind present (a paging file too small, then a failed CPU
/// allocation, ...), the last such line.
pub fn host_out_of_memory_line(lines: &[String]) -> Option<&String> {
    const KINDS: [&[&str]; 4] = [
        &["[WinError 1455]", "paging file is too small"],
        &["DefaultCPUAllocator: not enough memory", "[WinError 1450]"],
        &[
            "can't start new thread",
            "Cannot allocate memory",
            "[Errno 12]",
        ],
        &["is killed by signal: Killed"],
    ];
    KINDS
        .iter()
        .find_map(|pats| {
            lines
                .iter()
                .rev()
                .find(|l| pats.iter().any(|p| l.contains(p)))
        })
        // Python's own `MemoryError` only at the start of a line: `torch.OutOfMemoryError` (GPU)
        // contains it too.
        .or_else(|| {
            lines
                .iter()
                .rev()
                .find(|l| l.trim_start().starts_with("MemoryError"))
        })
}

/// The last line of `lines` that reports the GPU out of memory, if any. None when the host ran
/// out of memory too ([`host_out_of_memory_line`]): on Windows (WDDM) a CUDA allocation also
/// fails when the system commit is exhausted, with GPU memory free ("CUDA out of memory ...
/// 6.17 GiB is free"), and a smaller batch does not help there.
pub fn out_of_memory_line(lines: &[String]) -> Option<&String> {
    const PATTERNS: [&str; 5] = [
        "CUDA out of memory",
        "MPS backend out of memory",
        "torch.OutOfMemoryError",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "cudaErrorMemoryAllocation",
    ];
    if host_out_of_memory_line(lines).is_some() {
        return None;
    }
    lines
        .iter()
        .rev()
        .find(|l| PATTERNS.iter().any(|p| l.contains(p)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l(s: &str) -> Vec<String> {
        vec!["epoch 1".to_string(), s.to_string()]
    }

    /// QA-102: every GPU allocator failure counts, cuBLAS's and the CUDA runtime's included.
    #[test]
    fn every_gpu_allocation_failure_is_a_gpu_out_of_memory() {
        for line in [
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB",
            "RuntimeError: MPS backend out of memory (MPS allocated: 8.00 GB)",
            "RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
            "cudaErrorMemoryAllocation: out of memory",
        ] {
            assert_eq!(out_of_memory_line(&l(line)).map(String::as_str), Some(line));
        }
        assert_eq!(out_of_memory_line(&l("ValueError: bad box")), None);
    }

    /// The host running out wins: a CUDA failure beside it is not a GPU one.
    #[test]
    fn a_host_out_of_memory_is_not_a_gpu_one() {
        let lines = vec![
            "OSError: [WinError 1455] The paging file is too small".to_string(),
            "CUDA out of memory. 6.17 GiB is free".to_string(),
        ];
        assert!(host_out_of_memory_line(&lines).is_some());
        assert_eq!(out_of_memory_line(&lines), None);
    }
}
