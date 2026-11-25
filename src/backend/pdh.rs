//! One PDH (Windows Performance Data Helper) query holding one counter: open the query, add
//! the counter by its English path (so a localized Windows finds it too), take the first
//! collection (a rate needs two), close on failure and on drop. The one copy of that handle
//! handling, shared by [`super::memory::PageIns`] and the video-decode sampler
//! (`video_decode::imp::Query`), which keep only their read-out code. Not `Send`: the handles
//! stay on the thread that opened them.

use windows::Win32::System::Performance::{
    PDH_HCOUNTER, PDH_HQUERY, PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData,
    PdhOpenQueryW,
};
use windows::core::PCWSTR;

/// An open query with its counter. Closed once, on drop (closing it also removes the counter).
pub(crate) struct PdhQuery {
    query: PDH_HQUERY,
    counter: PDH_HCOUNTER,
}

impl std::fmt::Debug for PdhQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdhQuery").finish_non_exhaustive()
    }
}

impl PdhQuery {
    /// Open a live-data query, add the counter at `path` and collect once; None when PDH
    /// refuses any step (a broken performance counter registry, an unknown counter).
    ///
    /// # Safety
    /// `path` must point to a NUL-terminated wide string valid for the call (a `w!` literal).
    pub(crate) unsafe fn open(path: PCWSTR) -> Option<Self> {
        let mut query = PDH_HQUERY::default();
        // SAFETY: a null data source = live data; `query` is a live out-parameter written only
        // during the call.
        if unsafe { PdhOpenQueryW(PCWSTR::null(), 0, &mut query) } != 0 {
            return None;
        }
        let mut counter = PDH_HCOUNTER::default();
        // SAFETY: `query` was opened above; `path` is a NUL-terminated wide string (the
        // caller's contract); `counter` is a live out-parameter written only during the call.
        let added = unsafe { PdhAddEnglishCounterW(query, path, 0, &mut counter) };
        // SAFETY: `query` is open; collecting fills the counter's first raw values.
        if added != 0 || unsafe { PdhCollectQueryData(query) } != 0 {
            // SAFETY: `query` is open and closed once here; nothing uses it afterwards.
            unsafe { PdhCloseQuery(query) };
            return None;
        }
        Some(Self { query, counter })
    }

    /// Collect new raw values for the counter; false when PDH fails.
    pub(crate) fn collect(&self) -> bool {
        // SAFETY: `self.query` stays open until drop.
        unsafe { PdhCollectQueryData(self.query) == 0 }
    }

    /// The counter, valid while `self` lives (for `PdhGetFormattedCounter*` calls).
    pub(crate) fn counter(&self) -> PDH_HCOUNTER {
        self.counter
    }
}

impl Drop for PdhQuery {
    fn drop(&mut self) {
        // SAFETY: the query was opened in `open` and is closed once, here (closing it also
        // removes its counter).
        unsafe { PdhCloseQuery(self.query) };
    }
}
