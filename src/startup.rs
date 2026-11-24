//! Readiness state of a loaded model, shared between the loader/worker threads and HTTP handlers.
//!
//! An `Initializing` model may carry a detail text; [`StateHandle::set_waiting_for_download`]
//! marks the details of a download wait, so the pages label only those "Waiting for download"
//! and every other detail (a compile, a generation hand-over) "Loading" (QA-121).

use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelState {
    /// Model is being read/compiled, waits for a download, or is lazy and not yet requested.
    Initializing,
    Ready,
    Failed(String),
}

#[derive(Debug)]
struct Inner {
    state: ModelState,
    /// What an `Initializing` model is doing, e.g. "downloading OpenVINO runtime 42% (18/44 MB)".
    detail: Option<String>,
    /// `detail` is a download wait's ([`StateHandle::set_waiting_for_download`]).
    download: bool,
}

#[derive(Clone, Debug)]
pub struct StateHandle(Arc<RwLock<Inner>>);

impl Default for StateHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl StateHandle {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(Inner {
            state: ModelState::Initializing,
            detail: None,
            download: false,
        })))
    }

    pub fn get(&self) -> ModelState {
        self.0.read().map(|s| s.state.clone()).unwrap_or_else(|_| {
            ModelState::Failed(
                "the model state is unreadable after a crash in another thread; this is a \
                     bug; restart the service and report the log"
                    .into(),
            )
        })
    }

    /// Set the state; clears the detail.
    pub fn set(&self, s: ModelState) {
        if let Ok(mut g) = self.0.write() {
            g.state = s;
            g.detail = None;
            g.download = false;
        }
    }

    /// `Initializing` with a detail message (a compile, a hand-over, ...).
    pub fn set_initializing(&self, detail: impl Into<String>) {
        self.set_detail(detail.into(), false);
    }

    /// `Initializing` while the model's downloads install, with their progress (or the wait for
    /// a restart after a runtime installed): the only details shown as a download wait.
    pub fn set_waiting_for_download(&self, detail: impl Into<String>) {
        self.set_detail(detail.into(), true);
    }

    fn set_detail(&self, detail: String, download: bool) {
        if let Ok(mut g) = self.0.write() {
            g.state = ModelState::Initializing;
            g.detail = Some(detail);
            g.download = download;
        }
    }

    /// The detail of an `Initializing` model and whether it is a download wait, read together.
    pub fn detail_kind(&self) -> Option<(String, bool)> {
        self.0
            .read()
            .ok()
            .filter(|g| g.state == ModelState::Initializing)
            .and_then(|g| g.detail.clone().map(|d| (d, g.download)))
    }

    /// The detail of an `Initializing` model, if any.
    pub fn detail(&self) -> Option<String> {
        self.0
            .read()
            .ok()
            .filter(|g| g.state == ModelState::Initializing)
            .and_then(|g| g.detail.clone())
    }

    pub fn is_ready(&self) -> bool {
        self.get() == ModelState::Ready
    }

    /// "Initializing", "Initializing (downloading ...)", "Ready" or "Failed: `<reason>`".
    pub fn describe(&self) -> String {
        match (self.get(), self.detail()) {
            (ModelState::Initializing, Some(d)) => format!("Initializing ({d})"),
            (ModelState::Initializing, None) => "Initializing".into(),
            (ModelState::Ready, _) => "Ready".into(),
            (ModelState::Failed(m), _) => format!("Failed: {m}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_only_while_initializing() {
        let s = StateHandle::new();
        assert_eq!(s.describe(), "Initializing");
        s.set_waiting_for_download("downloading OpenVINO runtime 42% (18/44 MB)");
        assert_eq!(
            s.describe(),
            "Initializing (downloading OpenVINO runtime 42% (18/44 MB))"
        );
        s.set(ModelState::Ready);
        assert_eq!(s.detail(), None);
        assert_eq!(s.describe(), "Ready");
    }

    // QA-121: only a download wait's detail is one; any other detail is not.
    #[test]
    fn only_download_waits_are_tagged() {
        let s = StateHandle::new();
        assert_eq!(s.detail_kind(), None);
        s.set_waiting_for_download("downloading OpenVINO runtime 42%");
        assert_eq!(
            s.detail_kind(),
            Some(("downloading OpenVINO runtime 42%".into(), true))
        );
        s.set_initializing("the new generation is taking over");
        assert_eq!(
            s.detail_kind(),
            Some(("the new generation is taking over".into(), false))
        );
        s.set_waiting_for_download("waiting for a restart");
        s.set(ModelState::Initializing);
        assert_eq!(s.detail_kind(), None, "a new state clears the tag too");
        s.set_initializing("compiling on openvino:gpu");
        assert_eq!(s.detail_kind().map(|(_, d)| d), Some(false));
    }
}
