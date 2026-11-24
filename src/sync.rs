//! Poison-tolerant lock helpers, and [`nap`]: a thread's sleep that cancellation cuts short.
//!
//! Every `Mutex` / `RwLock` in this crate guards small state whose critical sections are short
//! and panic-free, so a lock poisoned by a panic elsewhere on the holding thread still guards
//! consistent data. These extension traits take the lock and recover from poisoning instead of
//! propagating the panic to every later user of the lock.
//!
//! Background threads (config watch, retention, auth flusher, sd-watchdog, resource watcher)
//! wait between ticks with [`nap`] instead of sleeping in short steps to notice cancellation:
//! the thread parks until its interval ends or the token is cancelled (from the token itself or
//! any parent), so an idle service wakes once per documented interval and still stops at once.

use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::Thread;
use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;

/// Poison-recovering `lock` for [`Mutex`].
pub trait LockExt<T: ?Sized> {
    /// Lock the mutex, blocking like [`Mutex::lock`]. A poisoned lock is recovered: the data is
    /// still consistent because every critical section here is short and panic-free.
    fn lock_recover(&self) -> MutexGuard<'_, T>;

    /// Try to lock without blocking: `None` when the lock is held elsewhere. A poisoned lock is
    /// recovered: the data is still consistent because every critical section here is short and
    /// panic-free.
    fn try_lock_recover(&self) -> Option<MutexGuard<'_, T>>;
}

impl<T: ?Sized> LockExt<T> for Mutex<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn try_lock_recover(&self) -> Option<MutexGuard<'_, T>> {
        match self.try_lock() {
            Ok(g) => Some(g),
            Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }
}

/// Poison-recovering `read` / `write` for [`RwLock`].
pub trait RwLockExt<T: ?Sized> {
    /// Take a shared read lock. A poisoned lock is recovered: the data is still consistent
    /// because every critical section here is short and panic-free.
    fn read_recover(&self) -> RwLockReadGuard<'_, T>;

    /// Take the exclusive write lock. A poisoned lock is recovered: the data is still consistent
    /// because every critical section here is short and panic-free.
    fn write_recover(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T: ?Sized> RwLockExt<T> for RwLock<T> {
    fn read_recover(&self) -> RwLockReadGuard<'_, T> {
        self.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write_recover(&self) -> RwLockWriteGuard<'_, T> {
        self.write().unwrap_or_else(|e| e.into_inner())
    }
}

/// Wakes the thread that parked in [`nap`].
struct Unpark(Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// Sleep `total` on this thread; `false` as soon as `stop` is (or already was) cancelled,
/// `true` when the whole time passed. The thread parks with no wake-up in between: the token's
/// cancel wakes it through the waker of `stop.cancelled()`, so no runtime is needed and a
/// parent token's cancel counts too.
pub fn nap(total: Duration, stop: &CancellationToken) -> bool {
    nap_counted(total, stop).0
}

/// [`nap`], also returning how many times the thread woke before it returned (tests assert an
/// idle nap wakes about once).
fn nap_counted(total: Duration, stop: &CancellationToken) -> (bool, u32) {
    let now = Instant::now();
    let end = now.checked_add(total);
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut cancelled = pin!(stop.cancelled());
    let mut wakes = 0u32;
    loop {
        if let Poll::Ready(()) = cancelled.as_mut().poll(&mut cx) {
            return (false, wakes);
        }
        let now = Instant::now();
        let left = match end {
            Some(end) if now >= end => return (true, wakes),
            Some(end) => end - now,
            // A total past the clock's range: wait for the cancel only.
            None => Duration::from_secs(3600),
        };
        std::thread::park_timeout(left);
        wakes = wakes.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nap_sleeps_its_time_with_one_wake() {
        let stop = CancellationToken::new();
        let t = Instant::now();
        let (whole, wakes) = nap_counted(Duration::from_millis(700), &stop);
        assert!(whole);
        assert!(t.elapsed() >= Duration::from_millis(700));
        // park_timeout may return early (spurious wake-ups are allowed), never in a 100 ms loop.
        assert!(wakes <= 3, "{wakes} wake-ups for one idle nap");
    }

    #[test]
    fn nap_returns_at_once_when_already_cancelled() {
        let stop = CancellationToken::new();
        stop.cancel();
        assert_eq!(nap_counted(Duration::from_secs(60), &stop), (false, 0));
    }

    #[test]
    fn a_cancel_from_another_thread_or_a_parent_wakes_the_nap() {
        for through_parent in [false, true] {
            let parent = CancellationToken::new();
            let child = parent.child_token();
            let canceller = if through_parent {
                parent.clone()
            } else {
                child.clone()
            };
            let h = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                canceller.cancel();
            });
            let t = Instant::now();
            assert!(!nap(Duration::from_secs(60), &child));
            assert!(t.elapsed() < Duration::from_secs(10));
            h.join().unwrap();
        }
    }

    #[test]
    fn a_huge_total_still_wakes_on_cancel() {
        let stop = CancellationToken::new();
        let c = stop.clone();
        let h = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            c.cancel();
        });
        assert!(!nap(Duration::MAX, &stop));
        h.join().unwrap();
    }

    #[test]
    fn mutex_recovers_from_poison() {
        let m = Arc::new(Mutex::new(1));
        let m2 = m.clone();
        let _ = std::thread::spawn(move || {
            let _g = m2.lock().unwrap();
            panic!("poison");
        })
        .join();
        assert!(m.is_poisoned());
        *m.lock_recover() += 1;
        assert_eq!(*m.lock_recover(), 2);
        assert_eq!(m.try_lock_recover().map(|g| *g), Some(2));
    }

    #[test]
    fn try_lock_recover_is_none_when_held() {
        let m = Mutex::new(0);
        let _g = m.lock_recover();
        assert!(m.try_lock_recover().is_none());
    }

    #[test]
    fn rwlock_recovers_from_poison() {
        let l = Arc::new(RwLock::new(String::from("a")));
        let l2 = l.clone();
        let _ = std::thread::spawn(move || {
            let _g = l2.write().unwrap();
            panic!("poison");
        })
        .join();
        assert!(l.is_poisoned());
        l.write_recover().push('b');
        assert_eq!(*l.read_recover(), "ab");
    }
}
