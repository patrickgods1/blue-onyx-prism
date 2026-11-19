//! The memory commit check before a job starts a Python model process (Windows): auto-label, the
//! overlap check, the dataset check (its fold training and its closer look) and an import's
//! SigLIP 2 selection. A job whose estimated commit ([`Need`]) does not fit what is free now plus
//! what the paging files may still grow by ([`Headroom`]) is refused with what happened, why and
//! what to do ([`problem`], pure), before any Python process is spawned.
//!
//! Why: Windows grows a paging file whose maximum is above its current size when the commit charge
//! nears the limit, in jumps of about 2 to 5.5 GB, until its drive is nearly full. The paging file
//! is usually on the system drive, which on a Blue Iris box often holds the recordings too: an
//! auto-label run (SAM 3 + RF-DETR L + Qwen3-VL 2B + SigLIP 2) grew `c:\pagefile.sys` to its 24 GB
//! maximum and left C: with 236 MB free (2026-11-18). So growth counts only while
//! [`PAGING_DISK_MARGIN_BYTES`] stay free on the file's drive, the rule `training::workers_for`
//! applies to data loader workers ([`growable_bytes`]).
//!
//! The rule: refused when `need + COMMIT_SLACK_BYTES > free commit + growable`. Only on Windows:
//! Linux overcommits and macOS grows its swap, so there is no limit to plan against ([`problem`]
//! takes the OS, so it is tested everywhere). A failed reading refuses nothing ([`read`] is None):
//! the check is a guard in front of jobs that ran without it, and a reading that failed says
//! nothing about the memory; refusing then would block every such job on that machine with a
//! message whose numbers are unknown and whose advice may not apply. The job then runs as it did
//! before the check, and a real shortage still ends it with WinError 1455 (as `workers_for` falls
//! back to the cores rule without a reading).
//!
//! The needs of each job are named constants beside the job, with where each figure came from:
//! [`crate::training::autolabel::memory::commit_need`], [`crate::training::overlaps::zoom::commit_need`],
//! [`crate::training::refine::commit_need`], [`crate::training::import::commit_need`].

use crate::backend::memory::{self, Commit, PAGING_DISK_MARGIN_BYTES, PagingFile, growable_bytes};

const GB: u64 = 1 << 30;

/// Commit kept spare on top of a job's estimate. Measured: an RF-DETR L + SigLIP 2 auto-label run
/// (about 4.6 GB, flat after loading) started with 5.05 GB of commit free (18.6 of 23.65 GB in use)
/// and Windows still grew the paging file by 2 GB, so 0.45 GB to spare was not enough (the load's
/// peak sits above its steady size, and Blue Iris and the service commit more while a job runs).
/// The 1 GB is an estimate above that.
pub const COMMIT_SLACK_BYTES: u64 = GB;

/// What a job will commit at its peak (Windows), with the words of its refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Need {
    /// The job, as a sentence's subject: "this auto-label run", "the overlap check".
    pub what: String,
    /// Estimated peak commit of its processes, bytes ([`COMMIT_SLACK_BYTES`] is added on top).
    pub bytes: u64,
    /// What it loads: "SAM 3 6.4 GB + RF-DETR L 2.6 GB".
    pub detail: String,
    /// A lighter run that needs less, as an action: "turn SAM 3 off". None: no lighter run.
    pub lighter: Option<String>,
}

/// A paging file's drive whose free space limits its growth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Drive {
    /// `C:`.
    pub name: String,
    /// Free bytes on it now.
    pub free: u64,
}

/// What a job can commit now (Windows): the free commit and how far the paging files may grow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Headroom {
    /// The commit limit and what is free of it now.
    pub commit: Commit,
    /// Bytes the paging files may still grow by while [`PAGING_DISK_MARGIN_BYTES`] stay free on
    /// each drive ([`growable_bytes`]).
    pub growable: u64,
    /// Bytes they could grow by to their maximum sizes, whatever the free space (`growable` <=
    /// this).
    pub room: u64,
    /// The paging files configured (0: paging is off).
    pub files: usize,
    /// The drives whose free space less the margin holds the growth below `room`.
    pub limited: Vec<Drive>,
    /// A growable file whose drive or free space could not be read: what it adds is unknown, so
    /// nothing is refused ([`problem`]).
    pub unread: bool,
}

impl Headroom {
    /// [`Headroom`] of `commit` on a machine with `total_phys` of RAM and the paging `files`, each
    /// on the volume root `volume_of` names (`C:\`) with `free_space` bytes free there (pure).
    pub fn of(
        commit: Commit,
        total_phys: u64,
        files: &[PagingFile],
        volume_of: impl Fn(&PagingFile) -> Option<String>,
        free_space: impl Fn(&str) -> Option<u64>,
    ) -> Self {
        let grow = |free: &dyn Fn(&str) -> Option<u64>| {
            growable_bytes(files, commit.limit, total_phys, &volume_of, free)
        };
        let room = grow(&|_| Some(u64::MAX));
        let growable = grow(&|r| free_space(r));
        let mut roots: Vec<String> = Vec::new();
        let mut unread = false;
        for f in files {
            match volume_of(f) {
                Some(r) => {
                    let r = r.to_ascii_uppercase();
                    if !roots.contains(&r) {
                        roots.push(r);
                    }
                }
                None => unread |= !f.fixed(),
            }
        }
        let mut limited = Vec::new();
        for root in roots {
            // Only this volume's free space applies: does it hold the growth below the maxima?
            let capped = grow(&|r| {
                if r.eq_ignore_ascii_case(&root) {
                    free_space(r)
                } else {
                    Some(u64::MAX)
                }
            });
            if capped >= room {
                continue;
            }
            match free_space(&root) {
                Some(free) => limited.push(Drive {
                    name: root.trim_end_matches('\\').to_string(),
                    free,
                }),
                None => unread = true,
            }
        }
        Self {
            commit,
            growable,
            room,
            files: files.len(),
            limited,
            unread,
        }
    }

    /// Commit a job can take now: the free commit plus the growth.
    pub fn total(&self) -> u64 {
        self.commit.available.saturating_add(self.growable)
    }
}

/// `1.0 GB`-style text of `bytes`.
fn gb(bytes: u64) -> String {
    format!("{:.1} GB", super::gib(bytes))
}

/// "C:", "C: and D:", "C:, D: and E:".
fn and_list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

/// Why `need` cannot start on `os` with `headroom`, or None: it fits (with
/// [`COMMIT_SLACK_BYTES`] to spare), the OS is not Windows, or the reading failed (None, or
/// [`Headroom::unread`]; see the module docs for why that refuses nothing). Pure.
///
/// The sentence says what the job needs, what is free, why the paging file cannot make up the
/// rest (a drive near its [`PAGING_DISK_MARGIN_BYTES`], a fixed or full-size file, no paging
/// file), and what to do, ending "then start the job again".
pub fn problem(os: &str, need: &Need, headroom: Option<&Headroom>) -> Option<String> {
    if os != "windows" {
        return None;
    }
    let h = headroom?;
    let want = need.bytes.saturating_add(COMMIT_SLACK_BYTES);
    if want <= h.total() || h.unread {
        return None;
    }
    let margin = PAGING_DISK_MARGIN_BYTES / GB;
    let mut actions: Vec<String> = Vec::new();
    let paging = if !h.limited.is_empty() {
        let names: Vec<String> = h.limited.iter().map(|d| d.name.clone()).collect();
        let (file, there) = if names.len() == 1 {
            ("the paging file", "there")
        } else {
            ("the paging files", "on each")
        };
        let frees: Vec<String> = h
            .limited
            .iter()
            .map(|d| {
                if names.len() == 1 {
                    gb(d.free)
                } else {
                    format!("{} {}", d.name, gb(d.free))
                }
            })
            .collect();
        let by = if h.growable > 0 {
            format!(" by only {} more", gb(h.growable))
        } else {
            " only".to_string()
        };
        let on = and_list(&names);
        actions.push(format!("free space on {on}"));
        actions.push(format!("put {file} on another drive"));
        format!(
            " and {file} on {on} can grow{by} while {margin} GB stays free {there} ({} free now)",
            and_list(&frees)
        )
    } else if h.files == 0 {
        actions.push("close programs that use memory".into());
        actions.push("turn on a paging file on a drive with room to spare".into());
        " and paging is off".to_string()
    } else if h.room == 0 {
        actions.push("close programs that use memory".into());
        actions
            .push("let Windows manage the paging file size on a drive with room to spare".into());
        " and the paging file cannot grow (a fixed size, or at its maximum)".to_string()
    } else {
        actions.push("close programs that use memory".into());
        actions.push("raise the paging file's maximum size on a drive with room to spare".into());
        format!(
            " and the paging file can grow by only {} more (its maximum size)",
            gb(h.room)
        )
    };
    actions.extend(need.lighter.clone());
    let mut first = and_or(&actions);
    if let Some(c) = first.get(..1).map(str::to_ascii_uppercase) {
        first.replace_range(..1, &c);
    }
    Some(format!(
        "{} needs about {} of memory commit ({}, {} kept spare); {} is free{paging}. {first}, \
         then start the job again.",
        need.what,
        gb(want),
        need.detail,
        gb(COMMIT_SLACK_BYTES),
        gb(h.commit.available),
    ))
}

/// "a", "a or b", "a, b or c".
fn and_or(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [init @ .., last] => format!("{}, or {last}", init.join(", ")),
    }
}

/// This machine's [`Headroom`] now: the commit (`GlobalMemoryStatusEx`), the registry's paging
/// files and each one's drive free space. None off Windows or when the commit, the RAM or the
/// registry cannot be read (nothing is refused then; see the module docs). Not cached.
pub fn read() -> Option<Headroom> {
    let commit = crate::system_info::commit()?;
    let physical = crate::system_info::physical()?;
    let (files, existing) = memory::paging_registry()?;
    let system_drive = std::env::var("SystemDrive").ok();
    Some(Headroom::of(
        commit,
        physical.total,
        &files,
        |f| memory::volume_root(f, &existing, system_drive.as_deref()),
        |root| fs4::available_space(root).ok(),
    ))
}

/// The reading jobs and previews use: [`read`], except in unit tests, where it is what the test's
/// thread set ([`tests_reading::set`], none by default), so no test depends on the host's memory.
pub fn now() -> Option<Headroom> {
    #[cfg(test)]
    {
        tests_reading::get()
    }
    #[cfg(not(test))]
    {
        read()
    }
}

/// [`problem`] of `need` with a reading of this machine (a job's [`now`] seam). A reading exists
/// only on Windows ([`read`] is None elsewhere, since `GlobalMemoryStatusEx` is the source), so a
/// reading is judged as Windows: a job test with a faked reading means the same on every OS.
pub fn refusal(need: &Need, reading: Option<&Headroom>) -> Option<String> {
    problem("windows", need, reading)
}

/// [`refusal`] of `need` on this machine now ([`now`]).
pub fn refusal_now(need: &Need) -> Option<String> {
    refusal(need, now().as_ref())
}

/// The reading [`now`] gives in unit tests: per thread, none by default.
#[cfg(test)]
pub(crate) mod tests_reading {
    use super::Headroom;
    use std::cell::RefCell;

    thread_local! {
        static READING: RefCell<Option<Headroom>> = const { RefCell::new(None) };
    }

    pub(crate) fn get() -> Option<Headroom> {
        READING.with(|r| r.borrow().clone())
    }

    /// Set (or clear) it for this thread.
    pub(crate) fn set(h: Option<Headroom>) {
        READING.with(|r| *r.borrow_mut() = h);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(crate) mod tests {
    use super::*;
    use crate::backend::memory::{parse_paging_files, volume_root};

    pub(crate) fn gbf(v: f64) -> u64 {
        (v * GB as f64) as u64
    }

    /// `files` on a machine with 15.9 GB of RAM, a commit limit of `limit_gb` with `free_gb` free,
    /// and `disk` free bytes per volume root.
    pub(crate) fn headroom(
        lines: &[&str],
        limit_gb: f64,
        free_gb: f64,
        disk: &[(&str, f64)],
    ) -> Headroom {
        let files = parse_paging_files(&lines.iter().map(|s| (*s).to_string()).collect::<Vec<_>>());
        Headroom::of(
            Commit {
                limit: gbf(limit_gb),
                available: gbf(free_gb),
            },
            gbf(15.9),
            &files,
            |f| volume_root(f, &[], Some("C:")),
            |root| disk.iter().find(|(r, _)| *r == root).map(|(_, b)| gbf(*b)),
        )
    }

    /// This box on 2026-11-18: 15.9 GB of RAM, `c:\pagefile.sys 2935 24576`, commit 23.65 GB
    /// limit with about 5.05 GB free (18.6 GB in use), 15.2 GB free on C:.
    pub(crate) fn this_box() -> Headroom {
        headroom(
            &[r"c:\pagefile.sys 2935 24576"],
            23.65,
            5.05,
            &[(r"C:\", 15.2)],
        )
    }

    fn need(what: &str, gb_need: f64) -> Need {
        Need {
            what: what.into(),
            bytes: gbf(gb_need),
            detail: format!("{what} {gb_need} GB"),
            lighter: None,
        }
    }

    #[test]
    fn this_box_cannot_grow_its_paging_file_and_says_why() {
        let h = this_box();
        // The file is 7.75 GB now (23.65 - 15.9) of 24: 16.25 GB of room, but C: has 15.2 GB free,
        // under the 16 GB margin: no growth.
        assert_eq!(h.growable, 0);
        assert_eq!(h.room, gbf(24.0) - (gbf(23.65) - gbf(15.9)));
        assert_eq!(
            h.limited,
            vec![Drive {
                name: "C:".into(),
                free: gbf(15.2)
            }]
        );
        assert!(!h.unread);
        let e = problem("windows", &need("this auto-label run", 9.0), Some(&h)).unwrap();
        assert_eq!(
            e,
            "this auto-label run needs about 10.0 GB of memory commit (this auto-label run 9 GB, \
             1.0 GB kept spare); 5.0 GB is free and the paging file on C: can grow only while 16 \
             GB stays free there (15.2 GB free now). Free space on C:, or put the paging file on \
             another drive, then start the job again."
        );
    }

    #[test]
    fn a_job_fits_the_free_commit_with_the_slack_to_spare() {
        let h = this_box();
        // 5.05 GB free: 4.0 + 1 fits, 4.1 + 1 does not.
        assert_eq!(problem("windows", &need("a", 4.0), Some(&h)), None);
        assert!(problem("windows", &need("a", 4.1), Some(&h)).is_some());
        // The measured RF-DETR L + SigLIP 2 run (4.6 GB) grew the paging file by 2 GB from this
        // state: refused.
        assert!(problem("windows", &need("a", 4.6), Some(&h)).is_some());
    }

    #[test]
    fn growth_counts_until_the_drive_margin() {
        // 19 GB free on C: keeps 16: 3 GB of growth.
        let h = headroom(
            &[r"c:\pagefile.sys 2935 24576"],
            23.65,
            5.05,
            &[(r"C:\", 19.0)],
        );
        assert_eq!(h.growable, gbf(19.0) - PAGING_DISK_MARGIN_BYTES);
        assert_eq!(problem("windows", &need("a", 7.0), Some(&h)), None);
        let e = problem("windows", &need("a", 7.1), Some(&h)).unwrap();
        assert!(
            e.contains(
                "and the paging file on C: can grow by only 3.0 GB more while 16 GB stays free \
                 there (19.0 GB free now)"
            ),
            "{e}"
        );
        // Plenty of disk: the maximum size limits it (7.6 GB now, 24 GB at most).
        let h = headroom(
            &[r"c:\pagefile.sys 2935 24576"],
            23.5,
            5.05,
            &[(r"C:\", 100.0)],
        );
        assert!(h.limited.is_empty());
        assert_eq!(h.growable, h.room);
        let e = problem("windows", &need("a", 30.0), Some(&h)).unwrap();
        assert!(
            e.contains("and the paging file can grow by only 16.4 GB more (its maximum size)"),
            "{e}"
        );
        assert!(e.contains("raise the paging file's maximum size"), "{e}");
    }

    #[test]
    fn a_fixed_paging_file_gives_no_growth() {
        let h = headroom(
            &[r"c:\pagefile.sys 5600 5600"],
            21.4,
            5.05,
            &[(r"C:\", 100.0)],
        );
        assert_eq!((h.growable, h.room), (0, 0));
        assert!(h.limited.is_empty());
        let e = problem("windows", &need("the overlap check", 4.6), Some(&h)).unwrap();
        assert!(
            e.contains(
                "5.0 GB is free and the paging file cannot grow (a fixed size, or at its maximum)"
            ),
            "{e}"
        );
        assert!(e.contains("Close programs that use memory"), "{e}");
        // No paging file at all.
        let h = headroom(&[], 15.9, 5.05, &[]);
        let e = problem("windows", &need("a", 9.0), Some(&h)).unwrap();
        assert!(e.contains("and paging is off"), "{e}");
    }

    #[test]
    fn two_drives_near_their_margin_are_both_named() {
        let h = headroom(
            &[r"c:\pagefile.sys 2048 8192", r"d:\pagefile.sys 2048 8192"],
            15.9 + 4.0,
            2.0,
            &[(r"C:\", 15.0), (r"D:\", 10.0)],
        );
        assert_eq!(h.limited.len(), 2);
        let e = problem("windows", &need("a", 5.0), Some(&h)).unwrap();
        assert!(
            e.contains(
                "the paging files on C: and D: can grow only while 16 GB stays free on each (C: \
                 15.0 GB and D: 10.0 GB free now). Free space on C: and D:, or put the paging \
                 files on another drive"
            ),
            "{e}"
        );
    }

    #[test]
    fn a_lighter_run_is_offered_last() {
        let mut n = need("this auto-label run", 9.0);
        n.lighter = Some("turn SAM 3 off".into());
        let e = problem("windows", &n, Some(&this_box())).unwrap();
        assert!(
            e.ends_with(
                "Free space on C:, put the paging file on another drive, or turn SAM 3 off, then \
                 start the job again."
            ),
            "{e}"
        );
    }

    /// A failed reading refuses nothing: no reading at all, or a growable file whose drive or
    /// free space cannot be read.
    #[test]
    fn a_missing_reading_refuses_nothing() {
        let n = need("a", 30.0);
        assert_eq!(problem("windows", &n, None), None);
        let unknown_disk = headroom(&[r"c:\pagefile.sys 2935 24576"], 23.65, 1.0, &[]);
        assert!(unknown_disk.unread, "{unknown_disk:?}");
        assert_eq!(problem("windows", &n, Some(&unknown_disk)), None);
        // A file with no drive (`?:` with neither ExistingPageFiles nor SystemDrive).
        let files = parse_paging_files(&[r"?:\pagefile.sys".to_string()]);
        let h = Headroom::of(
            Commit {
                limit: gbf(30.0),
                available: GB,
            },
            gbf(15.9),
            &files,
            |f| volume_root(f, &[], None),
            |_| Some(gbf(100.0)),
        );
        assert!(h.unread);
        assert_eq!(problem("windows", &n, Some(&h)), None);
        // A fixed file with an unreadable drive adds nothing either way: still judged.
        let fixed = headroom(&[r"c:\pagefile.sys 5600 5600"], 21.4, 1.0, &[]);
        assert!(!fixed.unread);
        assert!(problem("windows", &n, Some(&fixed)).is_some());
    }

    #[test]
    fn only_windows_is_checked() {
        let n = need("a", 30.0);
        let h = this_box();
        assert!(problem("windows", &n, Some(&h)).is_some());
        assert_eq!(problem("linux", &n, Some(&h)), None);
        assert_eq!(problem("macos", &n, Some(&h)), None);
    }

    /// The live reading: present on Windows, consistent; None elsewhere.
    #[test]
    fn read_is_windows_only_and_consistent() {
        let h = read();
        if cfg!(windows) {
            let h = h.unwrap();
            println!("{h:?}");
            assert!(h.growable <= h.room, "{h:?}");
            assert!(h.commit.available <= h.commit.limit, "{h:?}");
        } else {
            assert_eq!(h, None);
        }
        // Unit tests never see the host through `now`.
        assert_eq!(now(), None);
    }
}
