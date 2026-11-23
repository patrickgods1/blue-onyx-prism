//! Windows folder DACL inspection (audit 2026-11, RA-044): [`dacl_protected`] tells whether a
//! folder's DACL is protected (`SE_DACL_PROTECTED`: it inherits nothing from its parent), so
//! [`crate::private_dir`] can re-protect `auth/` when an earlier start crashed or failed between
//! creating it and restricting it. Lives under `backend/` because it needs `unsafe` FFI.
//!
//! Elsewhere (Unix) folder privacy is a mode, re-applied on every start, and this always answers
//! `Ok(true)` (nothing to repair here).

use anyhow::Result;
use std::path::Path;

/// Whether `dir`'s DACL is protected from inheritance (Windows: `GetNamedSecurityInfoW` with
/// `DACL_SECURITY_INFORMATION`, then `SE_DACL_PROTECTED` in the descriptor's control bits). An
/// error when the descriptor cannot be read (no such folder, no `READ_CONTROL` right). Always
/// `Ok(true)` off Windows.
pub fn dacl_protected(dir: &Path) -> Result<bool> {
    #[cfg(windows)]
    {
        windows_dacl_protected(dir)
    }
    #[cfg(not(windows))]
    {
        let _ = dir;
        Ok(true)
    }
}

/// A security descriptor `GetNamedSecurityInfoW` allocated, freed with `LocalFree` on drop.
#[cfg(windows)]
struct Descriptor(windows::Win32::Security::PSECURITY_DESCRIPTOR);

#[cfg(windows)]
impl Drop for Descriptor {
    fn drop(&mut self) {
        if self.0.is_invalid() {
            return;
        }
        // SAFETY: the pointer came from GetNamedSecurityInfoW, which documents LocalFree as its
        // release; it is freed exactly once (this value owns it) and not used afterwards.
        let _ = unsafe {
            windows::Win32::Foundation::LocalFree(Some(windows::Win32::Foundation::HLOCAL(
                self.0.0,
            )))
        };
    }
}

#[cfg(windows)]
fn windows_dacl_protected(dir: &Path) -> Result<bool> {
    use std::os::windows::ffi::OsStrExt;
    use windows::Win32::Foundation::ERROR_SUCCESS;
    use windows::Win32::Security::Authorization::{GetNamedSecurityInfoW, SE_FILE_OBJECT};
    use windows::Win32::Security::{
        DACL_SECURITY_INFORMATION, GetSecurityDescriptorControl, PSECURITY_DESCRIPTOR,
        SE_DACL_PROTECTED,
    };
    let wide: Vec<u16> = dir
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let mut sd = PSECURITY_DESCRIPTOR::default();
    // SAFETY: `wide` is a NUL-terminated UTF-16 path that outlives the call; `sd` is a live,
    // writable out pointer; the owner, group, DACL and SACL out pointers are null (allowed when
    // the descriptor itself is requested). On success `sd` owns a LocalAlloc'd descriptor, freed
    // by `Descriptor`'s drop.
    let rc = unsafe {
        GetNamedSecurityInfoW(
            windows::core::PCWSTR(wide.as_ptr()),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            None,
            None,
            None,
            None,
            &mut sd,
        )
    };
    if rc != ERROR_SUCCESS {
        anyhow::bail!(
            "reading the permissions of {} failed (Windows error {}); check that the folder \
             exists and that this account may read its permissions",
            dir.display(),
            rc.0
        );
    }
    let sd = Descriptor(sd);
    let (mut control, mut revision) = (0u16, 0u32);
    // SAFETY: `sd.0` is the valid descriptor returned above and stays allocated for the call;
    // `control` and `revision` are live, writable out values.
    unsafe { GetSecurityDescriptorControl(sd.0, &mut control, &mut revision) }.map_err(|e| {
        anyhow::anyhow!(
            "reading the permission flags of {} failed: {e}",
            dir.display()
        )
    })?;
    Ok(control & SE_DACL_PROTECTED.0 != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(windows))]
    #[test]
    fn off_windows_every_folder_counts_as_protected() {
        assert!(dacl_protected(&std::env::temp_dir()).unwrap());
    }

    /// A fresh folder in the temp dir inherits its parent's DACL; after `icacls /inheritance:r`
    /// it is protected. A missing folder is an error.
    #[cfg(windows)]
    #[test]
    fn a_folder_is_protected_after_inheritance_is_removed() {
        let dir = std::env::temp_dir().join(format!("bop_acl_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!dacl_protected(&dir).unwrap(), "a new folder inherits");
        let out = std::process::Command::new("icacls")
            .arg(&dir)
            .arg("/inheritance:d")
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        assert!(dacl_protected(&dir).unwrap());
        std::fs::remove_dir_all(&dir).ok();
        assert!(dacl_protected(&dir).is_err());
    }
}
