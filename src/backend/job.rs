//! Windows Job Object that kills a child process **tree** (`resources::export`, finding RS-02).
//!
//! `std::process::Child::kill` only ends the direct child; `uv` and PyTorch spawn helpers that
//! would survive a cancel, a stage timeout or a service stop. A [`Job`] with
//! `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` holds the child and everything it spawns: [`Job::kill`]
//! terminates them all, and dropping the job (or the service dying and the OS closing its
//! handles) does the same. Unix uses process groups instead and needs no FFI (see
//! `resources::export`). Windows 8+ allows nested jobs, so this works when the service itself
//! runs inside a job.

use anyhow::{Context, Result};
use std::os::windows::io::AsRawHandle;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
    SetInformationJobObject, TerminateJobObject,
};

/// An anonymous job object with kill-on-close.
#[derive(Debug)]
pub struct Job {
    handle: HANDLE,
}

// SAFETY: a job handle is a kernel object reference; it can be used and closed from any thread.
unsafe impl Send for Job {}
// SAFETY: see above; the handle is never mutated after creation.
unsafe impl Sync for Job {}

impl Job {
    /// Create a job whose processes are terminated when the last handle to it closes.
    pub fn new() -> Result<Self> {
        // SAFETY: no security attributes and no name; the returned handle is owned by `Job` and
        // closed in `Drop`.
        let handle = unsafe { CreateJobObjectW(None, windows::core::PCWSTR::null()) }
            .context("CreateJobObject")?;
        let job = Job { handle };
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let len = u32::try_from(std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>())
            .context("limit information size")?;
        // SAFETY: `info` is a properly initialized, repr(C) structure of the class and length
        // passed, alive for the duration of the call.
        unsafe {
            SetInformationJobObject(
                job.handle,
                JobObjectExtendedLimitInformation,
                std::ptr::from_ref(&info).cast(),
                len,
            )
        }
        .context("SetInformationJobObject(kill on close)")?;
        Ok(job)
    }

    /// Put `child` (and every process it will spawn) into the job. Call right after `spawn`;
    /// helpers the child starts before this call are not covered.
    pub fn assign(&self, child: &std::process::Child) -> Result<()> {
        let process = HANDLE(child.as_raw_handle().cast());
        // SAFETY: both handles are valid for the duration of the call: the job is owned by
        // `self` and the process handle is owned by `child`, which the caller keeps alive.
        unsafe { AssignProcessToJobObject(self.handle, process) }
            .context("AssignProcessToJobObject")
    }

    /// Terminate every process in the job (exit code 1).
    pub fn kill(&self) -> Result<()> {
        // SAFETY: the job handle is valid until `Drop`.
        unsafe { TerminateJobObject(self.handle, 1) }.context("TerminateJobObject")
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        // SAFETY: the handle was returned by CreateJobObjectW and is closed exactly once here.
        // Closing the last handle terminates the job's remaining processes (kill-on-close).
        let _ = unsafe { CloseHandle(self.handle) };
    }
}
