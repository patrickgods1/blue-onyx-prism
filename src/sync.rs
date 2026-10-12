//! Poison-tolerant lock helpers.
//!
//! Every `Mutex` / `RwLock` in this crate guards small state whose critical sections are short
//! and panic-free, so a lock poisoned by a panic elsewhere on the holding thread still guards
//! consistent data. These extension traits take the lock and recover from poisoning instead of
//! propagating the panic to every later user of the lock.

use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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
