//! The server-wide budget of detection upload bytes held in memory (M7). [`super::BODY_LIMIT`]
//! bounds one request and the in-flight cap the number of requests, but the cap grows with the
//! models' queues, so their product reaches several GiB. Every upload's bytes are charged here as
//! the multipart read receives them ([`Charge::add`]) and given back when the last reference to
//! them drops ([`charged_bytes`] wraps the image so its clones in the worker, the collector or the
//! Test page hold the charge). A read that ends early (a client hang-up, the read deadline, a
//! panic) drops its [`Charge`] and gives its bytes back with it.
//!
//! Blue Iris sends one JPEG of a few hundred kB per request: even a burst of every camera at once
//! stays far below [`UPLOAD_BUDGET`], so only an abusive or broken client reaches it.

use crate::metrics::RejectReason;
use bytes::{Bytes, BytesMut};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Upload bytes held in memory at once, across all detection requests.
pub(crate) const UPLOAD_BUDGET: usize = 1024 * 1024 * 1024;

/// The reason a refusal of this budget is counted under in `requests_rejected_total`.
pub(crate) const REJECT_REASON: RejectReason = RejectReason::UploadBudget;

/// A budget of bytes; [`UPLOADS`] is the one the detection routes use.
#[derive(Debug)]
pub(crate) struct UploadBudget {
    limit: usize,
    used: AtomicUsize,
    rejected: AtomicU64,
}

/// The process-wide budget of the detection routes.
pub(crate) static UPLOADS: UploadBudget = UploadBudget::new(UPLOAD_BUDGET);

impl UploadBudget {
    pub(crate) const fn new(limit: usize) -> Self {
        Self {
            limit,
            used: AtomicUsize::new(0),
            rejected: AtomicU64::new(0),
        }
    }

    /// Bytes charged now.
    pub(crate) fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    pub(crate) fn limit(&self) -> usize {
        self.limit
    }

    /// Requests refused by this budget so far.
    #[cfg(test)]
    pub(crate) fn rejected(&self) -> u64 {
        self.rejected.load(Ordering::Relaxed)
    }

    /// Count one refusal.
    pub(crate) fn note_rejected(&self) {
        self.rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether `n` more bytes fit now. Takes nothing: a check before a read, which a concurrent
    /// read may overtake; [`Charge::add`] is the bound.
    pub(crate) fn has_room(&self, n: usize) -> bool {
        self.used().saturating_add(n) <= self.limit
    }

    /// Take `n` bytes when they fit (all or nothing).
    fn try_take(&self, n: usize) -> bool {
        self.used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |u| {
                u.checked_add(n).filter(|&next| next <= self.limit)
            })
            .is_ok()
    }

    fn give_back(&self, n: usize) {
        if n > 0 {
            self.used.fetch_sub(n, Ordering::AcqRel);
        }
    }

    /// An empty charge against this budget.
    pub(crate) fn charge(&'static self) -> Charge {
        Charge {
            budget: self,
            held: 0,
        }
    }
}

/// Bytes taken from an [`UploadBudget`], given back when dropped.
#[derive(Debug)]
pub(crate) struct Charge {
    budget: &'static UploadBudget,
    held: usize,
}

impl Charge {
    /// Take `n` more bytes; false, taking nothing, when the budget has no room for them.
    #[must_use]
    pub(crate) fn add(&mut self, n: usize) -> bool {
        if !self.budget.try_take(n) {
            return false;
        }
        self.held = self.held.saturating_add(n);
        true
    }

    /// Bytes this charge holds.
    pub(crate) fn held(&self) -> usize {
        self.held
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.budget.give_back(self.held);
    }
}

/// The bytes of `buf` holding `charge` until the last clone of the returned `Bytes` drops.
pub(crate) fn charged_bytes(buf: BytesMut, charge: Charge) -> Bytes {
    if charge.held() == 0 {
        return buf.freeze();
    }
    Bytes::from_owner(Charged {
        bytes: buf.freeze(),
        _charge: charge,
    })
}

/// The owner behind [`charged_bytes`].
struct Charged {
    bytes: Bytes,
    _charge: Charge,
}

impl AsRef<[u8]> for Charged {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
pub(crate) fn leaked(limit: usize) -> &'static UploadBudget {
    Box::leak(Box::new(UploadBudget::new(limit)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn a_charge_takes_what_fits_and_gives_it_back_on_drop() {
        let b = leaked(100);
        let mut c = b.charge();
        assert!(c.add(60));
        assert!(c.add(40));
        assert_eq!(b.used(), 100);
        assert!(!c.add(1), "over the limit");
        assert_eq!(c.held(), 100, "a refused add takes nothing");
        let mut other = b.charge();
        assert!(!other.add(1));
        assert!(b.has_room(0) && !b.has_room(1));
        drop(c);
        assert_eq!(b.used(), 0);
        assert!(other.add(100));
        drop(other);
        assert_eq!(b.used(), 0);
        assert!(!b.charge().add(usize::MAX), "no overflow");
        assert_eq!(b.used(), 0);
    }

    #[test]
    fn charged_bytes_hold_the_charge_until_the_last_clone_drops() {
        let b = leaked(1000);
        let mut c = b.charge();
        assert!(c.add(5));
        let bytes = charged_bytes(BytesMut::from(&b"hello"[..]), c);
        assert_eq!(&bytes[..], b"hello");
        let clone = bytes.clone();
        let part = bytes.slice(1..3);
        drop(bytes);
        assert_eq!(b.used(), 5);
        drop(clone);
        assert_eq!(b.used(), 5, "a slice holds it too");
        assert_eq!(&part[..], b"el");
        drop(part);
        assert_eq!(b.used(), 0);
        // Nothing charged: plain bytes.
        let plain = charged_bytes(BytesMut::from(&b"x"[..]), b.charge());
        assert_eq!(&plain[..], b"x");
    }

    #[test]
    fn a_panic_gives_the_bytes_back() {
        let b = leaked(10);
        let r = std::panic::catch_unwind(|| {
            let mut c = b.charge();
            assert!(c.add(7));
            panic!("mid-read");
        });
        assert!(r.is_err());
        assert_eq!(b.used(), 0);
    }

    #[test]
    fn concurrent_charges_never_pass_the_limit() {
        let b = leaked(1000);
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(move || {
                    let mut held = Vec::new();
                    for _ in 0..500 {
                        let mut c = b.charge();
                        if c.add(30) {
                            held.push(c);
                        }
                        assert!(b.used() <= 1000);
                        if held.len() > 3 {
                            held.remove(0);
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(b.used(), 0);
    }
}
