//! Windows Job Object that kills a child process **tree** (`resources::export`, finding RS-02).
//!
//! `std::process::Child::kill` only ends the direct child; `uv` and PyTorch spawn helpers that
//! would survive a cancel, a stage timeout or a service stop. A [`Job`] with
//! `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` holds the child and everything it spawns: [`Job::kill`]
//! terminates them all, and dropping the job (or the service dying and the OS closing its
//! handles) does the same. Unix uses process groups instead and needs no FFI (see
//! `resources::export`). Windows 8+ allows nested jobs, so this works when the service itself
//! runs inside a job.
//!
//! [`Job::suspend`] pauses the whole tree (every thread of every process in the job, kept as
//! open handles in [`Suspended`]) and [`Suspended::resume`] (or dropping it) continues it: the
//! Training page's Pause / Resume. Windows has no documented "suspend a job" call, so the threads
//! are found with a ToolHelp snapshot.

use anyhow::{Context, Result};
use std::os::windows::io::AsRawHandle;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JOBOBJECT_BASIC_PROCESS_ID_LIST, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JobObjectBasicProcessIdList, JobObjectExtendedLimitInformation, QueryInformationJobObject,
    SetInformationJobObject, TerminateJobObject,
};
use windows::Win32::System::Threading::{
    OpenThread, ResumeThread, SuspendThread, THREAD_SUSPEND_RESUME,
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

impl Job {
    /// Ids of the processes in the job now.
    fn process_ids(&self) -> Result<Vec<u32>> {
        let word = std::mem::size_of::<usize>();
        // The id array starts after the two u32 counts (8 bytes on 32- and 64-bit Windows).
        let head = std::mem::offset_of!(JOBOBJECT_BASIC_PROCESS_ID_LIST, ProcessIdList) / word;
        let mut cap = 64usize;
        loop {
            // `usize` words: the alignment of the structure, whose ids are `usize` too.
            let mut buf = vec![0usize; head + cap];
            let len = u32::try_from(buf.len() * word).context("process list size")?;
            // SAFETY: `buf` is a zeroed, usize-aligned buffer of `len` bytes, large enough for
            // the structure header plus `cap` ids, alive for the duration of the call.
            let r = unsafe {
                QueryInformationJobObject(
                    Some(self.handle),
                    JobObjectBasicProcessIdList,
                    buf.as_mut_ptr().cast(),
                    len,
                    None,
                )
            };
            // SAFETY: the buffer starts with the structure's header (two u32), written by the
            // call or still zero.
            let list = unsafe { &*buf.as_ptr().cast::<JOBOBJECT_BASIC_PROCESS_ID_LIST>() };
            let assigned = list.NumberOfAssignedProcesses as usize;
            let listed = (list.NumberOfProcessIdsInList as usize).min(cap);
            if (r.is_err() || assigned > listed) && cap < 1 << 16 {
                // ERROR_MORE_DATA: more processes than room; ask again with more.
                cap = (assigned + 16).max(cap * 4);
                continue;
            }
            r.context("QueryInformationJobObject(process ids)")?;
            return Ok(buf
                .get(head..head + listed)
                .unwrap_or_default()
                .iter()
                .filter_map(|&id| u32::try_from(id).ok())
                .collect());
        }
    }

    /// Suspend every thread of every process in the job. Repeats the listing until a pass finds
    /// no new thread (a process that was starting while the first pass ran is caught by the
    /// next). The threads run again on [`Suspended::resume`] or when it is dropped.
    pub fn suspend(&self) -> Result<Suspended> {
        let mut held = Suspended {
            threads: Vec::new(),
        };
        let mut seen = std::collections::HashSet::new();
        for _ in 0..5 {
            let pids: std::collections::HashSet<u32> = self.process_ids()?.into_iter().collect();
            if pids.is_empty() {
                break;
            }
            let mut new = 0usize;
            // SAFETY: a thread snapshot of the whole system; the handle is closed below.
            let snap = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) }
                .context("CreateToolhelp32Snapshot(threads)")?;
            let mut entry = THREADENTRY32 {
                dwSize: u32::try_from(std::mem::size_of::<THREADENTRY32>()).unwrap_or(28),
                ..Default::default()
            };
            // SAFETY: `snap` is a valid snapshot handle and `entry` a THREADENTRY32 with
            // `dwSize` set, alive for each call.
            let mut more = unsafe { Thread32First(snap, &mut entry) }.is_ok();
            while more {
                if pids.contains(&entry.th32OwnerProcessID) && seen.insert(entry.th32ThreadID) {
                    // SAFETY: plain thread open by id; a failure (the thread exited) is skipped.
                    let opened =
                        unsafe { OpenThread(THREAD_SUSPEND_RESUME, false, entry.th32ThreadID) };
                    if let Ok(h) = opened {
                        // SAFETY: `h` is a thread handle opened with THREAD_SUSPEND_RESUME.
                        if unsafe { SuspendThread(h) } == u32::MAX {
                            // SAFETY: `h` was opened above and is closed once.
                            let _ = unsafe { CloseHandle(h) };
                        } else {
                            held.threads.push(h);
                            new += 1;
                        }
                    }
                }
                // SAFETY: as for Thread32First.
                more = unsafe { Thread32Next(snap, &mut entry) }.is_ok();
            }
            // SAFETY: the snapshot handle is closed exactly once.
            let _ = unsafe { CloseHandle(snap) };
            if new == 0 {
                break;
            }
        }
        Ok(held)
    }
}

/// Threads suspended by [`Job::suspend`], held open (so a thread id cannot be reused while it is
/// suspended); resumed by [`Self::resume`] or on drop.
#[derive(Debug)]
pub struct Suspended {
    threads: Vec<HANDLE>,
}

impl Suspended {
    /// Number of threads suspended.
    pub fn len(&self) -> usize {
        self.threads.len()
    }

    /// No thread was suspended (the job had no process).
    pub fn is_empty(&self) -> bool {
        self.threads.is_empty()
    }

    /// Let the threads run again.
    pub fn resume(self) {
        drop(self);
    }
}

impl Drop for Suspended {
    fn drop(&mut self) {
        for h in self.threads.drain(..) {
            // SAFETY: each handle was opened with THREAD_SUSPEND_RESUME and suspended once by
            // `Job::suspend`; it is resumed and closed exactly once here.
            unsafe {
                ResumeThread(h);
                let _ = CloseHandle(h);
            }
        }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        // SAFETY: the handle was returned by CreateJobObjectW and is closed exactly once here.
        // Closing the last handle terminates the job's remaining processes (kill-on-close).
        let _ = unsafe { CloseHandle(self.handle) };
    }
}
