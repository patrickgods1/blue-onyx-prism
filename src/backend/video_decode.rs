//! GPU video-decode load during a benchmark's timed phase: on the primary box Blue Iris decodes
//! its cameras with Intel Quick Sync on the same UHD 630 the model runs on, so a run measured
//! beside it is not a run on an idle GPU. Windows only: the PDH counter
//! `\GPU Engine(*engtype_VideoDecode)\Utilization Percentage` (English name, so a
//! localized Windows finds it too), sampled once a second ([`SAMPLE_EVERY`]) on its own thread
//! ([`Sampler`]). Elsewhere, or when the counter cannot be opened, [`Sampler::start`] is None and
//! nothing is recorded; a failed reading is skipped, never an error of the run.
//!
//! The counter has one instance per process and engine
//! (`pid_1234_luid_0x00000000_0x0000D1A5_phys_0_eng_1_engtype_VideoDecode`): one sample is the
//! busiest engine, each engine's load summed over the processes using it ([`busiest_engine`],
//! capped at 100% like Task Manager's). Those parts are pure and tested on every OS.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

/// Time between two samples.
pub const SAMPLE_EVERY: Duration = Duration::from_secs(1);

/// The video-decode load of one timed phase (percent of the busiest decode engine).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VideoDecode {
    pub mean: f64,
    pub max: f64,
    /// Samples taken ([`SAMPLE_EVERY`] apart).
    #[serde(default)]
    pub samples: usize,
}

impl VideoDecode {
    /// Mean and max of `samples` (percent); None without any.
    pub fn of(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let max = samples.iter().copied().fold(0.0, f64::max);
        Some(Self {
            mean: samples.iter().sum::<f64>() / samples.len() as f64,
            max,
            samples: samples.len(),
        })
    }
}

/// One decode engine of one GPU: the adapter LUID (both halves, lowercase), the physical
/// adapter and the engine index of an instance name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EngineKey {
    pub luid: String,
    pub phys: u32,
    pub eng: u32,
}

/// The engine of an instance name (`pid_.._luid_0x.._0x.._phys_N_eng_M_engtype_VideoDecode`);
/// None for a name without a LUID, physical adapter and engine index.
pub fn engine_key(instance: &str) -> Option<EngineKey> {
    let lower = instance.to_ascii_lowercase();
    let after = |key: &str| lower.find(key).map(|i| &lower[i + key.len()..]);
    let luid = after("luid_")?;
    let mut parts = luid.splitn(3, '_');
    let (hi, lo) = (parts.next()?, parts.next()?);
    let hex = |s: &str| {
        s.starts_with("0x") && s.len() > 2 && s[2..].chars().all(|c| c.is_ascii_hexdigit())
    };
    if !hex(hi) || !hex(lo) {
        return None;
    }
    let number = |key: &str| -> Option<u32> {
        let rest = after(key)?;
        let end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        rest[..end].parse().ok()
    };
    Some(EngineKey {
        luid: format!("{hi}_{lo}"),
        phys: number("_phys_")?,
        eng: number("_eng_")?,
    })
}

/// One sample from the instances' values (percent): each engine's load summed over its
/// processes and capped at 100, then the busiest engine. Instances whose name names no engine
/// and values that are not finite are skipped; 0 when nothing is left (no process decodes).
pub fn busiest_engine<'a>(items: impl IntoIterator<Item = (&'a str, f64)>) -> f64 {
    let mut per: BTreeMap<EngineKey, f64> = BTreeMap::new();
    for (name, v) in items {
        let Some(k) = engine_key(name) else { continue };
        if v.is_finite() && v >= 0.0 {
            *per.entry(k).or_default() += v;
        }
    }
    per.values().map(|v| v.min(100.0)).fold(0.0, f64::max)
}

/// Samples the video-decode load on its own thread until [`Sampler::finish`].
#[derive(Debug)]
pub struct Sampler {
    stop: crossbeam_channel::Sender<()>,
    thread: std::thread::JoinHandle<Vec<f64>>,
}

impl Sampler {
    /// Start sampling; None off Windows or when the counter cannot be opened (never an error).
    pub fn start() -> Option<Self> {
        let (ready_tx, ready_rx) = crossbeam_channel::bounded::<bool>(1);
        let (stop, stop_rx) = crossbeam_channel::bounded::<()>(1);
        let thread = std::thread::Builder::new()
            .name("bench-vdecode".into())
            .spawn(move || {
                let Some(mut q) = imp::Query::open() else {
                    let _ = ready_tx.send(false);
                    return Vec::new();
                };
                let _ = ready_tx.send(true);
                let started = std::time::Instant::now();
                let mut samples = Vec::new();
                loop {
                    match stop_rx.recv_timeout(SAMPLE_EVERY) {
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            samples.extend(q.sample());
                        }
                        _ => {
                            // A phase shorter than one interval still gets its one reading
                            // (the counter's rate needs some time after the first collect).
                            if samples.is_empty() && started.elapsed() >= MIN_LAST_READING {
                                samples.extend(q.sample());
                            }
                            return samples;
                        }
                    }
                }
            })
            .ok()?;
        if ready_rx.recv().unwrap_or(false) {
            Some(Self { stop, thread })
        } else {
            let _ = thread.join();
            None
        }
    }

    /// Stop sampling; the phase's load, None when no reading succeeded.
    pub fn finish(self) -> Option<VideoDecode> {
        let _ = self.stop.send(());
        VideoDecode::of(&self.thread.join().ok()?)
    }
}

/// A phase must last this long for its single reading at [`Sampler::finish`] (a rate over a few
/// milliseconds is noise).
const MIN_LAST_READING: Duration = Duration::from_millis(200);

#[cfg(windows)]
mod imp {
    use windows::Win32::System::Performance::{
        PDH_CSTATUS_NEW_DATA, PDH_CSTATUS_VALID_DATA, PDH_FMT, PDH_FMT_COUNTERVALUE_ITEM_W,
        PDH_FMT_DOUBLE, PDH_HCOUNTER, PDH_HQUERY, PDH_MORE_DATA, PDH_NO_DATA,
        PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW,
        PdhOpenQueryW,
    };
    use windows::core::{PCWSTR, w};

    /// `PDH_FMT_NOCAP100`: values above 100 are not capped per instance (the sum is).
    const PDH_FMT_NOCAP100: u32 = 0x0000_8000;

    /// The open query and its wildcard counter (owned by the sampling thread: the handles are
    /// not `Send`).
    pub(super) struct Query {
        query: PDH_HQUERY,
        counter: PDH_HCOUNTER,
    }

    impl Query {
        /// Open the query, add the counter and take the first collection (a rate needs two).
        pub(super) fn open() -> Option<Self> {
            let mut query = PDH_HQUERY::default();
            // SAFETY: a null data source = live data; `query` is a live out-parameter written
            // only during the call.
            if unsafe { PdhOpenQueryW(PCWSTR::null(), 0, &mut query) } != 0 {
                return None;
            }
            let mut counter = PDH_HCOUNTER::default();
            // SAFETY: `query` was opened above; the path is a NUL-terminated wide string literal;
            // `counter` is a live out-parameter written only during the call.
            let added = unsafe {
                PdhAddEnglishCounterW(
                    query,
                    w!(r"\GPU Engine(*engtype_VideoDecode)\Utilization Percentage"),
                    0,
                    &mut counter,
                )
            };
            // SAFETY: `query` is open; this fills the counter's first raw values.
            if added != 0 || unsafe { PdhCollectQueryData(query) } != 0 {
                // SAFETY: `query` is open and closed once here; nothing uses it afterwards.
                unsafe { PdhCloseQuery(query) };
                return None;
            }
            Some(Self { query, counter })
        }

        /// One sample ([`super::busiest_engine`]) since the previous collection; None when the
        /// reading fails.
        pub(super) fn sample(&mut self) -> Option<f64> {
            // SAFETY: `self.query` stays open until drop.
            if unsafe { PdhCollectQueryData(self.query) } != 0 {
                return None;
            }
            let format = PDH_FMT(PDH_FMT_DOUBLE.0 | PDH_FMT_NOCAP100);
            let (mut bytes, mut count) = (0u32, 0u32);
            // SAFETY: `self.counter` belongs to the open query; a null buffer with size 0 asks
            // for the size needed, written to the two live out-parameters.
            let rc = unsafe {
                PdhGetFormattedCounterArrayW(self.counter, format, &mut bytes, &mut count, None)
            };
            if rc == PDH_NO_DATA || (rc == 0 && count == 0) {
                // No process has a decode engine instance: nothing decodes.
                return Some(0.0);
            }
            if rc != PDH_MORE_DATA || bytes == 0 {
                return None;
            }
            // u64 words keep the buffer aligned for the items (pointers and doubles).
            let mut buf = vec![0u64; (bytes as usize).div_ceil(8)];
            let items = buf.as_mut_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>();
            // SAFETY: `buf` holds at least `bytes` bytes, aligned to 8, alive for the call; PDH
            // writes `count` items followed by their name strings into it.
            let rc = unsafe {
                PdhGetFormattedCounterArrayW(
                    self.counter,
                    format,
                    &mut bytes,
                    &mut count,
                    Some(items),
                )
            };
            if rc != 0 {
                return None;
            }
            let n = (count as usize)
                .min(buf.len() * 8 / std::mem::size_of::<PDH_FMT_COUNTERVALUE_ITEM_W>().max(1));
            // SAFETY: PDH wrote `count` initialized items at the start of `buf` (bounded by the
            // buffer's size above); `buf` outlives the slice.
            let slice = unsafe { std::slice::from_raw_parts(items, n) };
            let mut values: Vec<(String, f64)> = Vec::with_capacity(n);
            for item in slice {
                if !matches!(
                    item.FmtValue.CStatus,
                    PDH_CSTATUS_VALID_DATA | PDH_CSTATUS_NEW_DATA
                ) || item.szName.is_null()
                {
                    continue;
                }
                // SAFETY: `szName` points to a NUL-terminated wide string inside `buf`.
                let Ok(name) = (unsafe { item.szName.to_string() }) else {
                    continue;
                };
                // SAFETY: PDH_FMT_DOUBLE asked for the `doubleValue` member, which PDH filled.
                let v = unsafe { item.FmtValue.Anonymous.doubleValue };
                values.push((name, v));
            }
            Some(super::busiest_engine(
                values.iter().map(|(n, v)| (n.as_str(), *v)),
            ))
        }
    }

    impl Drop for Query {
        fn drop(&mut self) {
            // SAFETY: the query was opened in `open` and is closed once, here (closing it also
            // removes its counter).
            unsafe { PdhCloseQuery(self.query) };
        }
    }
}

#[cfg(not(windows))]
mod imp {
    /// No video-decode counter off Windows.
    pub(super) struct Query;

    impl Query {
        pub(super) fn open() -> Option<Self> {
            None
        }

        pub(super) fn sample(&mut self) -> Option<f64> {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_names_give_their_engine() {
        let k = engine_key("pid_1234_luid_0x00000000_0x0000D1A5_phys_0_eng_1_engtype_VideoDecode")
            .unwrap();
        assert_eq!(
            k,
            EngineKey {
                luid: "0x00000000_0x0000d1a5".into(),
                phys: 0,
                eng: 1
            }
        );
        // Another process, same engine: the same key (any case).
        assert_eq!(
            engine_key("PID_99_LUID_0X00000000_0X0000D1A5_PHYS_0_ENG_1_ENGTYPE_VIDEODECODE"),
            Some(k.clone())
        );
        assert_eq!(
            engine_key("pid_99_luid_0x00000000_0x0000d1a5_phys_0_eng_1_engtype_VideoDecode"),
            Some(k)
        );
        for bad in [
            "_Total",
            "pid_1_luid_0x0_phys_0_eng_1",
            "pid_1_luid_zz_0x1_phys_0_eng_1_engtype_VideoDecode",
            "pid_1_luid_0x1_0x2_phys_x_eng_1_engtype_VideoDecode",
            "pid_1_luid_0x1_0x2_phys_0_engtype_VideoDecode",
        ] {
            assert_eq!(engine_key(bad), None, "{bad}");
        }
    }

    /// Each engine's load is summed over its processes (capped at 100) and the busiest engine
    /// is the sample: two processes on engine 1 of the iGPU (20 + 15) beat one at 30 on its
    /// engine 2; another GPU counts as its own engines.
    #[test]
    fn a_sample_is_the_busiest_engine_summed_over_processes() {
        let n = |pid: u32, luid: &str, eng: u32| {
            format!("pid_{pid}_luid_0x00000000_{luid}_phys_0_eng_{eng}_engtype_VideoDecode")
        };
        let items = [
            (n(1, "0x0000d1a5", 1), 20.0),
            (n(2, "0x0000d1a5", 1), 15.0),
            (n(3, "0x0000d1a5", 2), 30.0),
            (n(4, "0x0000beef", 1), 31.0),
            ("_Total".to_string(), 99.0),
            (n(5, "0x0000d1a5", 2), f64::NAN),
        ];
        let v = busiest_engine(items.iter().map(|(a, b)| (a.as_str(), *b)));
        assert!((v - 35.0).abs() < 1e-9, "{v}");
        let over = [(n(1, "0x1", 0), 70.0), (n(2, "0x1", 0), 60.0)];
        assert_eq!(
            busiest_engine(over.iter().map(|(a, b)| (a.as_str(), *b))),
            100.0
        );
        assert_eq!(busiest_engine(std::iter::empty()), 0.0);
    }

    #[test]
    fn mean_and_max_of_the_samples() {
        assert_eq!(VideoDecode::of(&[]), None);
        let d = VideoDecode::of(&[10.0, 40.0, 25.0]).unwrap();
        assert_eq!((d.mean, d.max, d.samples), (25.0, 40.0, 3));
    }

    /// Off Windows nothing is sampled (and nothing fails).
    #[cfg(not(windows))]
    #[test]
    fn no_sampler_off_windows() {
        assert!(Sampler::start().is_none());
    }
}
