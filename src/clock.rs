//! The time zone of everything a person reads: log timestamps, file and folder names, and the
//! times shown on pages, in the CLI and in reports (config key `timezone`, [`Zone`]). Values
//! stored inside records (`created`, `reviewed_at`, `timestamp`, ...) stay UTC and are converted
//! only when shown; the Blue Iris API's `timestampUTC` stays UTC.
//!
//! Two layers, std + chrono only (no other module of this crate):
//! - **Pure helpers**: methods on [`Zone`] that take the instant ([`Zone::display`],
//!   [`Zone::day`], [`Zone::frame_stamp`], ...) and [`parse_frame_stamp`]. Anything that is
//!   tested calls these with an explicit zone.
//! - **The process-wide zone**: [`set`] / [`change`] / [`zone`], and wrappers named like the
//!   methods (`clock::display(t)` is `clock::zone().display(t)`) for code that shows a time.
//!   Every binary sets it from the config before logging starts; a config save or a restart
//!   that changes `timezone` calls [`change`] (logged at info).
//!
//! Formats, for 2026-11-16 22:02:11.583 UTC in `America/Los_Angeles` (PST, -08:00):
//!
//! | helper | text |
//! |---|---|
//! | [`Zone::log_stamp`] | `2026-11-16T14:02:11.583-08:00` (`...Z` when the zone is UTC) |
//! | [`Zone::display`] | `2026-11-16 14:02 PST` |
//! | [`Zone::display_date`] / [`Zone::day`] | `2026-11-16` |
//! | [`Zone::name_stamp`] | `20261116-140211` |
//! | [`Zone::frame_stamp`] | `20261116T140211583-0800` (`p0530` for +05:30, `Z` at offset 0) |
//! | [`Zone::browser_name`] | `America/Los_Angeles` |
//!
//! A frame stamp writes a positive offset with `p`, never `+`: `collect::layout::stem_ok`
//! allows only `[A-Za-z0-9_-]` in a stem, and `+` is a space in a URL query.
//!
//! Every instant argument is `impl Into<DateTime<Utc>>`, so a `SystemTime` (chrono's `From`)
//! works as well as a `DateTime<Utc>` or a `DateTime<FixedOffset>`.
//!
//! **Tests and the global.** [`zone`] is one value for the whole process, shared by every test
//! thread. A test never asserts text that depends on it (call the pure helpers with a zone
//! instead); a test that has to set it holds `clock::TEST_LOCK` (unit tests of this crate) and
//! puts the old value back.

use chrono::{
    DateTime, FixedOffset, LocalResult, NaiveDate, NaiveDateTime, Offset, TimeDelta, TimeZone, Utc,
};
use std::sync::{OnceLock, PoisonError, RwLock};

/// A time zone as the config stores it (`timezone`): `"utc"`, `"local"` (the computer's zone,
/// the default), an IANA name (`"America/Los_Angeles"`) or a fixed offset (`"+08:00"`).
/// `Display` writes the config form back; `FromStr` / serde read it (an unknown value is an
/// error naming the key).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Zone {
    /// UTC.
    Utc,
    /// The computer's time zone (chrono's `Local`; follows a change of the system zone).
    #[default]
    Local,
    /// An IANA zone (follows its daylight saving rules).
    Named(chrono_tz::Tz),
    /// A fixed offset from UTC (no daylight saving).
    Fixed(FixedOffset),
}

/// `+08:00` / `-05:30` / `+00:00` of an offset (seconds dropped; no zone in use has them today).
fn offset_text(o: FixedOffset) -> String {
    let secs = o.local_minus_utc();
    let (sign, a) = if secs < 0 { ('-', -secs) } else { ('+', secs) };
    format!("{sign}{:02}:{:02}", a / 3600, a % 3600 / 60)
}

/// The characters [`Zone::browser_name`] may hold (IANA names, `UTC`, offsets).
fn browser_safe(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '/' | '+' | ':' | '-')
}

/// The computer's IANA zone (`iana-time-zone`), looked up once per process. None when the OS
/// does not say or chrono-tz does not know the name.
fn local_iana() -> Option<chrono_tz::Tz> {
    static LOCAL: OnceLock<Option<chrono_tz::Tz>> = OnceLock::new();
    *LOCAL.get_or_init(|| {
        iana_time_zone::get_timezone()
            .ok()
            .and_then(|name| name.trim().parse::<chrono_tz::Tz>().ok())
    })
}

/// [`local_iana`] when it agrees with chrono's `Local` at `t` (a system zone changed since the
/// lookup does not, and is then shown by its offset).
fn local_iana_at(t: DateTime<Utc>) -> Option<chrono_tz::Tz> {
    let tz = local_iana()?;
    let local = chrono::Local.offset_from_utc_datetime(&t.naive_utc()).fix();
    (tz.offset_from_utc_datetime(&t.naive_utc()).fix() == local).then_some(tz)
}

/// The instant of wall time `naive` in `tz`: the earlier one at a fold (the hour repeated when
/// clocks go back), and in a gap (the hour skipped when clocks go forward) the wall time read
/// with the offset before the change, i.e. moved forward by the gap (02:30 on a spring-forward
/// night in Los Angeles is 03:30 PDT). None only out of chrono's range.
fn wall_to_utc<T: TimeZone>(tz: &T, naive: NaiveDateTime) -> Option<DateTime<Utc>> {
    match tz.from_local_datetime(&naive) {
        LocalResult::Single(t) => Some(t.with_timezone(&Utc)),
        LocalResult::Ambiguous(a, b) => Some(a.with_timezone(&Utc).min(b.with_timezone(&Utc))),
        LocalResult::None => {
            // No zone changes its offset twice within a day, so the offset a day earlier is the
            // one before the gap.
            let before = tz
                .offset_from_utc_datetime(&naive.checked_sub_signed(TimeDelta::days(1))?)
                .fix();
            let utc = naive
                .checked_sub_signed(TimeDelta::seconds(i64::from(before.local_minus_utc())))?;
            Some(utc.and_utc())
        }
    }
}

impl Zone {
    /// The offset from UTC in effect at `t`.
    ///
    /// ```
    /// # use blue_onyx_prism::clock::Zone;
    /// let la: Zone = "America/Los_Angeles".parse().unwrap();
    /// let t = chrono::DateTime::parse_from_rfc3339("2026-11-16T22:02:11Z").unwrap();
    /// assert_eq!(la.offset_at(t).local_minus_utc(), -8 * 3600);
    /// ```
    pub fn offset_at(self, t: impl Into<DateTime<Utc>>) -> FixedOffset {
        let t: DateTime<Utc> = t.into();
        let naive = t.naive_utc();
        match self {
            Zone::Utc => Utc.fix(),
            Zone::Local => chrono::Local.offset_from_utc_datetime(&naive).fix(),
            Zone::Named(tz) => tz.offset_from_utc_datetime(&naive).fix(),
            Zone::Fixed(o) => o,
        }
    }

    /// The wall-clock time at `t` in this zone.
    pub fn wall(self, t: impl Into<DateTime<Utc>>) -> NaiveDateTime {
        let t: DateTime<Utc> = t.into();
        t.with_timezone(&self.offset_at(t)).naive_local()
    }

    /// What follows a time in [`Self::display`]: `UTC`, the zone's abbreviation for an IANA zone
    /// (`PST`, `CEST`, `IST`) and for `local` when the computer's IANA zone is known, else the
    /// offset (`+08:00`; also for IANA zones whose abbreviation is only a number, such as
    /// `America/Sao_Paulo`'s `-03`).
    pub fn abbreviation(self, t: impl Into<DateTime<Utc>>) -> String {
        let t: DateTime<Utc> = t.into();
        let named = |tz: chrono_tz::Tz| {
            let a = t.with_timezone(&tz).format("%Z").to_string();
            let numeric = a.is_empty() || a.starts_with(['+', '-']);
            (!numeric).then_some(a)
        };
        let abbr = match self {
            Zone::Utc => Some("UTC".to_string()),
            Zone::Named(tz) => named(tz),
            Zone::Local => local_iana_at(t).and_then(named),
            Zone::Fixed(_) => None,
        };
        abbr.unwrap_or_else(|| offset_text(self.offset_at(t)))
    }

    /// The log timestamp of `t`: RFC 3339 with milliseconds, `...Z` for [`Zone::Utc`], else the
    /// numeric offset.
    ///
    /// ```
    /// # use blue_onyx_prism::clock::Zone;
    /// let t = chrono::DateTime::parse_from_rfc3339("2026-10-16T00:54:59.580Z").unwrap();
    /// assert_eq!(Zone::Utc.log_stamp(t), "2026-10-16T00:54:59.580Z");
    /// let la: Zone = "America/Los_Angeles".parse().unwrap();
    /// assert_eq!(la.log_stamp(t), "2026-10-15T17:54:59.580-07:00");
    /// ```
    pub fn log_stamp(self, t: impl Into<DateTime<Utc>>) -> String {
        use chrono::SecondsFormat::Millis;
        let t: DateTime<Utc> = t.into();
        match self {
            Zone::Utc => t.to_rfc3339_opts(Millis, true),
            _ => t
                .with_timezone(&self.offset_at(t))
                .to_rfc3339_opts(Millis, false),
        }
    }

    /// `t` for people: wall time to the minute and the [`Self::abbreviation`].
    ///
    /// ```
    /// # use blue_onyx_prism::clock::Zone;
    /// let t = chrono::DateTime::parse_from_rfc3339("2026-11-16T22:02:11Z").unwrap();
    /// let la: Zone = "America/Los_Angeles".parse().unwrap();
    /// assert_eq!(la.display(t), "2026-11-16 14:02 PST");
    /// assert_eq!(Zone::Utc.display(t), "2026-11-16 22:02 UTC");
    /// assert_eq!("+08:00".parse::<Zone>().unwrap().display(t), "2026-11-17 06:02 +08:00");
    /// ```
    pub fn display(self, t: impl Into<DateTime<Utc>>) -> String {
        let t: DateTime<Utc> = t.into();
        format!(
            "{} {}",
            self.wall(t).format("%Y-%m-%d %H:%M"),
            self.abbreviation(t)
        )
    }

    /// A stored time (RFC 3339, UTC or any offset) as [`Self::display`] shows it; `raw` itself
    /// when it is not one. Stored record timestamps stay UTC; pages, the CLI and reports show this.
    ///
    /// ```
    /// # use blue_onyx_prism::clock::Zone;
    /// let la: Zone = "America/Los_Angeles".parse().unwrap();
    /// assert_eq!(la.display_stored("2026-11-16T22:02:11.583Z"), "2026-11-16 14:02 PST");
    /// assert_eq!(la.display_stored("yesterday"), "yesterday");
    /// ```
    pub fn display_stored(self, raw: &str) -> String {
        DateTime::parse_from_rfc3339(raw).map_or_else(|_| raw.to_string(), |t| self.display(t))
    }

    /// `secs` (unix seconds, as `users.json` stores them) as [`Self::display`] shows it; the
    /// number itself when chrono cannot represent it.
    ///
    /// ```
    /// # use blue_onyx_prism::clock::Zone;
    /// let la: Zone = "America/Los_Angeles".parse().unwrap();
    /// assert_eq!(la.display_unix(1_763_330_531), "2025-11-16 14:02 PST");
    /// assert_eq!(Zone::Utc.display_unix(0), "1970-01-01 00:00 UTC");
    /// assert_eq!(Zone::Utc.display_unix(u64::MAX), u64::MAX.to_string());
    /// ```
    pub fn display_unix(self, secs: u64) -> String {
        i64::try_from(secs)
            .ok()
            .and_then(|s| DateTime::<Utc>::from_timestamp(s, 0))
            .map_or_else(|| secs.to_string(), |t| self.display(t))
    }

    /// The date of `t` in this zone, `YYYY-MM-DD`.
    ///
    /// ```
    /// # use blue_onyx_prism::clock::Zone;
    /// let t = chrono::DateTime::parse_from_rfc3339("2026-11-17T03:00:00Z").unwrap();
    /// let la: Zone = "America/Los_Angeles".parse().unwrap();
    /// assert_eq!(la.display_date(t), "2026-11-16");
    /// ```
    pub fn display_date(self, t: impl Into<DateTime<Utc>>) -> String {
        self.day(t).format("%Y-%m-%d").to_string()
    }

    /// The calendar day of `t` in this zone (day folders, "today", log files).
    pub fn day(self, t: impl Into<DateTime<Utc>>) -> NaiveDate {
        self.wall(t).date()
    }

    /// Wall time for names people read (run folders, reports, zips): `%Y%m%d-%H%M%S`, no zone.
    ///
    /// ```
    /// # use blue_onyx_prism::clock::Zone;
    /// let t = chrono::DateTime::parse_from_rfc3339("2026-11-16T22:02:11Z").unwrap();
    /// let la: Zone = "America/Los_Angeles".parse().unwrap();
    /// assert_eq!(la.name_stamp(t), "20261116-140211");
    /// ```
    pub fn name_stamp(self, t: impl Into<DateTime<Utc>>) -> String {
        self.wall(t).format("%Y%m%d-%H%M%S").to_string()
    }

    /// The timestamp at the start of a frame's file name: wall time with milliseconds and the
    /// offset, `20261116T140211583-0800`, `p0530` for a positive offset, `Z` at offset 0 (the
    /// old UTC form `20261116T220211583Z`). An offset with seconds (only historic local mean
    /// times) is written in UTC, so [`parse_frame_stamp`] always gives `t` back to the
    /// millisecond.
    ///
    /// ```
    /// # use blue_onyx_prism::clock::{Zone, parse_frame_stamp};
    /// let t = chrono::DateTime::parse_from_rfc3339("2026-11-16T22:02:11.583Z").unwrap();
    /// let la: Zone = "America/Los_Angeles".parse().unwrap();
    /// assert_eq!(la.frame_stamp(t), "20261116T140211583-0800");
    /// assert_eq!("Asia/Kolkata".parse::<Zone>().unwrap().frame_stamp(t), "20261117T033211583p0530");
    /// assert_eq!(Zone::Utc.frame_stamp(t), "20261116T220211583Z");
    /// assert_eq!(parse_frame_stamp("20261116T140211583-0800_ab12"), Some(t.to_utc()));
    /// ```
    pub fn frame_stamp(self, t: impl Into<DateTime<Utc>>) -> String {
        let t: DateTime<Utc> = t.into();
        let secs = self.offset_at(t).local_minus_utc();
        if secs == 0 || secs % 60 != 0 {
            return t.format("%Y%m%dT%H%M%S%3fZ").to_string();
        }
        let (sign, a) = if secs < 0 { ('-', -secs) } else { ('p', secs) };
        format!(
            "{}{sign}{:02}{:02}",
            self.wall(t).format("%Y%m%dT%H%M%S%3f"),
            a / 3600,
            a % 3600 / 60
        )
    }

    /// The instant a wall time in this zone names (a Blue Iris clip name, a date typed by a
    /// person). At a fold (clocks go back) the earlier instant; in a gap (clocks go forward) the
    /// wall time moved forward by the gap. None only outside chrono's range.
    ///
    /// ```
    /// # use blue_onyx_prism::clock::Zone;
    /// let la: Zone = "America/Los_Angeles".parse().unwrap();
    /// let wall = chrono::NaiveDate::from_ymd_opt(2026, 11, 1).unwrap().and_hms_opt(1, 30, 0).unwrap();
    /// // 01:30 happens twice that night: PDT first.
    /// assert_eq!(la.parse_wall(wall).unwrap().to_rfc3339(), "2026-11-01T08:30:00+00:00");
    /// ```
    pub fn parse_wall(self, naive: NaiveDateTime) -> Option<DateTime<Utc>> {
        match self {
            Zone::Utc => Some(naive.and_utc()),
            Zone::Local => wall_to_utc(&chrono::Local, naive),
            Zone::Named(tz) => wall_to_utc(&tz, naive),
            Zone::Fixed(o) => wall_to_utc(&o, naive),
        }
    }

    /// The zone for the browser (`<html data-tz>`): the IANA name of a named zone and of
    /// `local` when the computer's is known (and agrees with its current offset), `UTC`, else
    /// the current offset (`+08:00`). Only `[A-Za-z0-9_/+:-]`.
    ///
    /// ```
    /// # use blue_onyx_prism::clock::Zone;
    /// assert_eq!("Europe/Berlin".parse::<Zone>().unwrap().browser_name(), "Europe/Berlin");
    /// assert_eq!(Zone::Utc.browser_name(), "UTC");
    /// assert_eq!("-05:30".parse::<Zone>().unwrap().browser_name(), "-05:30");
    /// ```
    pub fn browser_name(self) -> String {
        let name = match self {
            Zone::Utc => "UTC".to_string(),
            Zone::Named(tz) => tz.name().to_string(),
            Zone::Fixed(o) => offset_text(o),
            Zone::Local => {
                let now = Utc::now();
                match local_iana_at(now) {
                    Some(tz) => tz.name().to_string(),
                    None => offset_text(self.offset_at(now)),
                }
            }
        };
        name.chars().filter(|&c| browser_safe(c)).collect()
    }
}

impl Zone {
    /// The letter abbreviations an IANA zone uses over the next year, for the browser
    /// (`<html data-tz-abbr>`): `offset-minutes:ABBR` pairs joined by `,` (`60:CET,120:CEST`),
    /// first one seen per offset, numeric abbreviations (`-03`) left out. The page's script
    /// (`assets/time.js`) looks up the offset of the instant it shows, so it prints what
    /// [`Self::abbreviation`] prints; `Intl` has no such names for most zones outside the US.
    /// Empty for `UTC`, fixed offsets and a `local` zone whose IANA name is unknown. Only
    /// `[A-Za-z0-9+:,-]`.
    ///
    /// ```
    /// # use blue_onyx_prism::clock::Zone;
    /// let now = chrono::DateTime::parse_from_rfc3339("2026-11-16T22:02:11Z").unwrap();
    /// let berlin: Zone = "Europe/Berlin".parse().unwrap();
    /// assert_eq!(berlin.abbreviation_map(now), "60:CET,120:CEST");
    /// assert_eq!(Zone::Utc.abbreviation_map(now), "");
    /// ```
    pub fn abbreviation_map(self, now: impl Into<DateTime<Utc>>) -> String {
        let now: DateTime<Utc> = now.into();
        let tz = match self {
            Zone::Named(tz) => Some(tz),
            Zone::Local => local_iana_at(now),
            Zone::Utc | Zone::Fixed(_) => None,
        };
        let Some(tz) = tz else {
            return String::new();
        };
        let mut seen: Vec<(i32, String)> = Vec::new();
        for month in 0..13_i64 {
            let t = now + TimeDelta::days(30 * month);
            let abbr = t.with_timezone(&tz).format("%Z").to_string();
            if abbr.is_empty() || abbr.starts_with(['+', '-']) || !abbr.chars().all(browser_safe) {
                continue;
            }
            let minutes = Zone::Named(tz).offset_at(t).local_minus_utc() / 60;
            if !seen.iter().any(|(m, _)| *m == minutes) {
                seen.push((minutes, abbr));
            }
        }
        seen.sort_by_key(|(m, _)| *m);
        seen.iter()
            .map(|(m, a)| format!("{m}:{a}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// [`Zone::abbreviation_map`] of the process-wide zone, now.
pub fn abbreviation_map() -> String {
    zone().abbreviation_map(Utc::now())
}

/// The instant at the start of `s` when it begins with a frame stamp ([`Zone::frame_stamp`]):
/// `YYYYMMDDTHHMMSSmmm` followed by `Z`, `-hhmm`, `phhmm` (or `+hhmm`), then the end or a
/// character that is not a letter or digit (stems are `<stamp>_<id>`). Old UTC stems
/// (`20261022T123456789Z_...`) parse too. None for anything else.
///
/// ```
/// # use blue_onyx_prism::clock::parse_frame_stamp;
/// let old = parse_frame_stamp("20261022T123456789Z_cam1").unwrap();
/// assert_eq!(old.to_rfc3339(), "2026-10-22T12:34:56.789+00:00");
/// let east = parse_frame_stamp("20261022T180456789p0530").unwrap();
/// assert_eq!(east, old);
/// assert_eq!(parse_frame_stamp("20261022T123456789"), None);
/// ```
pub fn parse_frame_stamp(s: &str) -> Option<DateTime<Utc>> {
    let b = s.as_bytes();
    let num = |from: usize, len: usize| -> Option<u32> {
        let digits = b.get(from..from + len)?;
        digits.iter().try_fold(0u32, |n, d| {
            d.is_ascii_digit().then(|| n * 10 + u32::from(d - b'0'))
        })
    };
    if b.get(8) != Some(&b'T') {
        return None;
    }
    let date = NaiveDate::from_ymd_opt(i32::try_from(num(0, 4)?).ok()?, num(4, 2)?, num(6, 2)?)?;
    let wall = date.and_hms_milli_opt(num(9, 2)?, num(11, 2)?, num(13, 2)?, num(15, 3)?)?;
    let (offset_secs, end) = match b.get(18)? {
        b'Z' => (0i64, 19),
        sign @ (b'-' | b'p' | b'+') => {
            let (h, m) = (num(19, 2)?, num(21, 2)?);
            if h > 23 || m > 59 {
                return None;
            }
            let secs = i64::from(h * 3600 + m * 60);
            (if *sign == b'-' { -secs } else { secs }, 23)
        }
        _ => return None,
    };
    if b.get(end).is_some_and(u8::is_ascii_alphanumeric) {
        return None;
    }
    Some(
        wall.checked_sub_signed(TimeDelta::seconds(offset_secs))?
            .and_utc(),
    )
}

impl std::fmt::Display for Zone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Zone::Utc => f.write_str("utc"),
            Zone::Local => f.write_str("local"),
            Zone::Named(tz) => f.write_str(tz.name()),
            Zone::Fixed(o) => f.write_str(&offset_text(*o)),
        }
    }
}

/// Why a `timezone` value was refused (names the key and the accepted forms).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseZoneError(String);

impl std::fmt::Display for ParseZoneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "timezone: '{}' is not a time zone. Use \"utc\", \"local\" (this computer's zone), \
             an IANA name such as \"America/Los_Angeles\" or a fixed offset such as \"+08:00\" \
             or \"-05:30\"; fix the value in the config file or on the Config page",
            self.0
        )
    }
}

impl std::error::Error for ParseZoneError {}

impl std::str::FromStr for Zone {
    type Err = ParseZoneError;
    fn from_str(s: &str) -> Result<Self, ParseZoneError> {
        let t = s.trim();
        let bad = || ParseZoneError(t.to_string());
        match t.to_ascii_lowercase().as_str() {
            "utc" => return Ok(Zone::Utc),
            "local" => return Ok(Zone::Local),
            _ => {}
        }
        if let Some(sign @ ('+' | '-')) = t.chars().next() {
            let digits = t.get(1..).unwrap_or_default().as_bytes();
            let &[h1, h2, b':', m1, m2] = digits else {
                return Err(bad());
            };
            if ![h1, h2, m1, m2].iter().all(u8::is_ascii_digit) {
                return Err(bad());
            }
            let two = |a: u8, b: u8| i32::from(a - b'0') * 10 + i32::from(b - b'0');
            let (h, m) = (two(h1, h2), two(m1, m2));
            if h > 23 || m > 59 {
                return Err(bad());
            }
            let secs = (h * 3600 + m * 60) * if sign == '-' { -1 } else { 1 };
            return FixedOffset::east_opt(secs).map(Zone::Fixed).ok_or_else(bad);
        }
        t.parse::<chrono_tz::Tz>()
            .map(Zone::Named)
            .map_err(|_| bad())
    }
}

impl TryFrom<String> for Zone {
    type Error = ParseZoneError;
    fn try_from(s: String) -> Result<Self, ParseZoneError> {
        s.parse()
    }
}

impl serde::Serialize for Zone {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> serde::Deserialize<'de> for Zone {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------------------------
// The process-wide zone
// ---------------------------------------------------------------------------------------------

/// The zone of this process ([`zone`]); `local` until a config sets it.
static CURRENT: RwLock<Zone> = RwLock::new(Zone::Local);

/// Held by unit tests that set the process-wide zone (they put the old one back).
#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The process-wide zone now (the config's `timezone`; `local` before one is set).
pub fn zone() -> Zone {
    *CURRENT.read().unwrap_or_else(PoisonError::into_inner)
}

/// Make `z` the process-wide zone, silently (at start, before logging); returns the previous.
pub fn set(z: Zone) -> Zone {
    std::mem::replace(
        &mut *CURRENT.write().unwrap_or_else(PoisonError::into_inner),
        z,
    )
}

/// [`set`] for a running process: a different zone is logged at info ("time zone changed from
/// X to Y", stamped in the new one). Returns whether it changed.
pub fn change(z: Zone) -> bool {
    let from = set(z);
    if from == z {
        return false;
    }
    tracing::info!(from = %from, to = %z, "time zone changed from {from} to {z}");
    true
}

/// [`Zone::log_stamp`] in the process-wide zone.
pub fn log_stamp(t: impl Into<DateTime<Utc>>) -> String {
    zone().log_stamp(t)
}

/// [`Zone::display`] in the process-wide zone.
pub fn display(t: impl Into<DateTime<Utc>>) -> String {
    zone().display(t)
}

/// [`Zone::display_stored`] in the process-wide zone.
pub fn display_stored(raw: &str) -> String {
    zone().display_stored(raw)
}

/// [`Zone::display_date`] in the process-wide zone.
pub fn display_date(t: impl Into<DateTime<Utc>>) -> String {
    zone().display_date(t)
}

/// [`Zone::day`] in the process-wide zone.
pub fn day(t: impl Into<DateTime<Utc>>) -> NaiveDate {
    zone().day(t)
}

/// [`Zone::name_stamp`] in the process-wide zone.
pub fn name_stamp(t: impl Into<DateTime<Utc>>) -> String {
    zone().name_stamp(t)
}

/// [`Zone::frame_stamp`] in the process-wide zone.
pub fn frame_stamp(t: impl Into<DateTime<Utc>>) -> String {
    zone().frame_stamp(t)
}

/// [`Zone::parse_wall`] in the process-wide zone.
pub fn parse_wall(naive: NaiveDateTime) -> Option<DateTime<Utc>> {
    zone().parse_wall(naive)
}

/// [`Zone::browser_name`] of the process-wide zone.
pub fn browser_name() -> String {
    zone().browser_name()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(s: &str) -> Zone {
        s.parse().unwrap()
    }

    fn at(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().to_utc()
    }

    fn wall(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, mo, d)
            .unwrap()
            .and_hms_opt(h, mi, 0)
            .unwrap()
    }

    #[test]
    fn parses_every_form() {
        assert_eq!(z(" UTC "), Zone::Utc);
        assert_eq!(z("Local"), Zone::Local);
        let la = z("America/Los_Angeles");
        assert!(matches!(la, Zone::Named(_)));
        assert_eq!(la.to_string(), "America/Los_Angeles");
        for (s, shown) in [
            ("+08:00", "+08:00"),
            ("-05:30", "-05:30"),
            ("+00:00", "+00:00"),
        ] {
            assert_eq!(z(s).to_string(), shown);
        }
        for bad in [
            "",
            "mars",
            "+8",
            "+25:00",
            "+08:60",
            "08:00",
            "+0800",
            "Nowhere/City",
        ] {
            let e = bad.parse::<Zone>().unwrap_err().to_string();
            assert!(e.contains("timezone") && e.contains("IANA"), "{bad}: {e}");
            assert!(!e.contains("log_timezone"), "{e}");
        }
        assert_eq!(Zone::default(), Zone::Local);
    }

    #[test]
    fn serde_round_trips_the_config_form() {
        for s in ["utc", "local", "Europe/Berlin", "-05:30"] {
            let json = serde_json::to_string(&z(s)).unwrap();
            assert_eq!(json, format!("\"{s}\""));
            assert_eq!(serde_json::from_str::<Zone>(&json).unwrap(), z(s));
        }
        let e = serde_json::from_str::<Zone>("\"mars\"")
            .unwrap_err()
            .to_string();
        assert!(e.contains("timezone") && e.contains("mars"), "{e}");
    }

    #[test]
    fn log_stamp_of_a_fixed_instant() {
        let t = at("2026-10-16T00:54:59.580Z");
        let f = |s: &str| z(s).log_stamp(t);
        assert_eq!(f("utc"), "2026-10-16T00:54:59.580Z");
        assert_eq!(f("+08:00"), "2026-10-16T08:54:59.580+08:00");
        assert_eq!(f("-05:30"), "2026-10-15T19:24:59.580-05:30");
        // DST: PDT (-07:00) in October, PST (-08:00) in January.
        assert_eq!(f("America/Los_Angeles"), "2026-10-15T17:54:59.580-07:00");
        let winter = at("2026-01-16T00:00:00Z");
        assert_eq!(
            z("America/Los_Angeles").log_stamp(winter),
            "2026-01-15T16:00:00.000-08:00"
        );
        assert!(DateTime::parse_from_rfc3339(&f("local")).is_ok());
        // A SystemTime works as well.
        let sys = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_millis(1500);
        assert_eq!(Zone::Utc.log_stamp(sys), "1970-01-01T00:00:01.500Z");
    }

    #[test]
    fn display_names_the_zone() {
        let t = at("2026-11-16T22:02:11.583Z");
        assert_eq!(z("America/Los_Angeles").display(t), "2026-11-16 14:02 PST");
        let summer = at("2026-07-16T22:02:11Z");
        assert_eq!(
            z("America/Los_Angeles").display(summer),
            "2026-07-16 15:02 PDT"
        );
        assert_eq!(z("Europe/Berlin").display(summer), "2026-07-17 00:02 CEST");
        assert_eq!(z("Asia/Kolkata").display(t), "2026-11-17 03:32 IST");
        assert_eq!(Zone::Utc.display(t), "2026-11-16 22:02 UTC");
        assert_eq!(z("+08:00").display(t), "2026-11-17 06:02 +08:00");
        assert_eq!(z("-05:30").display(t), "2026-11-16 16:32 -05:30");
        // A zone whose abbreviation is only a number shows the offset in the usual form.
        assert_eq!(z("America/Sao_Paulo").display(t), "2026-11-16 19:02 -03:00");
        // Local: wall time plus an abbreviation of letters or an offset.
        let local = Zone::Local.display(t);
        let (_, abbr) = local.rsplit_once(' ').unwrap();
        assert!(
            abbr.chars().all(|c| c.is_ascii_alphanumeric())
                || (abbr.len() == 6 && abbr.starts_with(['+', '-'])),
            "{local}"
        );
        assert_eq!(z("Europe/Berlin").display_date(t), "2026-11-16");
        assert_eq!(z("Asia/Tokyo").display_date(t), "2026-11-17");
    }

    #[test]
    fn day_at_the_midnight_edges() {
        let la = z("America/Los_Angeles");
        let d = |y, m, d| NaiveDate::from_ymd_opt(y, m, d).unwrap();
        // 00:00 PST = 08:00 UTC.
        assert_eq!(la.day(at("2026-11-17T07:59:59.999Z")), d(2026, 11, 16));
        assert_eq!(la.day(at("2026-11-17T08:00:00Z")), d(2026, 11, 17));
        // 00:00 IST = 18:30 UTC the day before.
        let ist = z("Asia/Kolkata");
        assert_eq!(ist.day(at("2026-11-16T18:29:59Z")), d(2026, 11, 16));
        assert_eq!(ist.day(at("2026-11-16T18:30:00Z")), d(2026, 11, 17));
        assert_eq!(Zone::Utc.day(at("2026-11-16T23:59:59Z")), d(2026, 11, 16));
        assert_eq!(z("+14:00").day(at("2026-11-16T10:00:00Z")), d(2026, 11, 17));
        assert_eq!(z("-12:00").day(at("2026-11-16T11:59:59Z")), d(2026, 11, 15));
    }

    #[test]
    fn name_stamp_is_wall_time() {
        let t = at("2026-11-16T22:02:11.583Z");
        assert_eq!(z("America/Los_Angeles").name_stamp(t), "20261116-140211");
        assert_eq!(Zone::Utc.name_stamp(t), "20261116-220211");
        assert_eq!(z("+05:30").name_stamp(t), "20261117-033211");
    }

    #[test]
    fn frame_stamps_round_trip() {
        let t = at("2026-11-16T22:02:11.583Z");
        // UTC: exactly the form collector stems had before the zone setting (`...123Z`).
        assert_eq!(Zone::Utc.frame_stamp(t), "20261116T220211583Z");
        assert_eq!(
            Zone::Utc.frame_stamp(t),
            t.format("%Y%m%dT%H%M%S%3fZ").to_string()
        );
        assert_eq!(
            z("America/Los_Angeles").frame_stamp(t),
            "20261116T140211583-0800"
        );
        assert_eq!(z("+05:30").frame_stamp(t), "20261117T033211583p0530");
        assert_eq!(
            z("Asia/Kathmandu").frame_stamp(t),
            "20261117T034711583p0545"
        );
        // Offset 0 is written `Z` (London in winter).
        assert_eq!(z("Europe/London").frame_stamp(t), "20261116T220211583Z");
        let zones = [
            "utc",
            "local",
            "America/Los_Angeles",
            "+05:30",
            "-05:30",
            "-00:30",
            "Asia/Kathmandu",
            "Pacific/Chatham",
            "Europe/London",
            "+14:00",
            "-12:00",
        ];
        let instants = [
            "2026-11-16T22:02:11.583Z",
            "2026-01-01T00:00:00.000Z",
            "2026-12-31T23:59:59.999Z",
            // Los Angeles: the fold (01:00-02:00 twice) and the gap night.
            "2026-11-01T08:30:00.001Z",
            "2026-11-01T09:30:00.002Z",
            "2026-03-08T09:59:59.999Z",
            "2026-03-08T10:00:00.000Z",
            "1999-02-28T12:00:00.500Z",
        ];
        for zone in zones {
            for i in instants {
                let t = at(i);
                let s = z(zone).frame_stamp(t);
                // Allowed in a stem (`collect::layout::stem_ok`) and safe in a URL.
                assert!(
                    s.chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')),
                    "{zone} {i}: {s}"
                );
                assert_eq!(parse_frame_stamp(&s), Some(t), "{zone} {i}: {s}");
                assert_eq!(
                    parse_frame_stamp(&format!("{s}_a1b2c3")),
                    Some(t),
                    "{zone} {i}"
                );
            }
        }
        // The two 01:30s of the fold night have different stamps.
        let first = z("America/Los_Angeles").frame_stamp(at("2026-11-01T08:30:00Z"));
        let second = z("America/Los_Angeles").frame_stamp(at("2026-11-01T09:30:00Z"));
        assert_eq!(first, "20261101T013000000-0700");
        assert_eq!(second, "20261101T013000000-0800");
    }

    #[test]
    fn parse_frame_stamp_reads_old_stems_and_refuses_the_rest() {
        let old = parse_frame_stamp("20261022T123456789Z_cam1-abc").unwrap();
        assert_eq!(old, at("2026-10-22T12:34:56.789Z"));
        assert_eq!(parse_frame_stamp("20261022T123456789Z"), Some(old));
        assert_eq!(parse_frame_stamp("20261022T183456789+0600"), Some(old));
        assert_eq!(parse_frame_stamp("20261022T063456789-0600"), Some(old));
        for bad in [
            "",
            "2026",
            "20261022T123456789",
            "20261022 123456789Z",
            "20261022T12345678Z",
            "20261022T123456789X",
            "20261022T123456789-08",
            "20261022T123456789-0860",
            "20261022T123456789-2500",
            "20261022T123456789Z1",
            "20261022T123456789-08001",
            "20261322T123456789Z",
            "20260230T123456789Z",
            "20261022T243456789Z",
            "2026102xT123456789Z",
            "-2026102T123456789Z",
            "frame_20261022T123456789Z",
            "20261022T123456789Zé",
        ] {
            let got = parse_frame_stamp(bad);
            // A non-ASCII character after the stamp ends it like `_`.
            if bad.ends_with('é') {
                assert_eq!(got, Some(old), "{bad}");
            } else {
                assert_eq!(got, None, "{bad}");
            }
        }
    }

    #[test]
    fn parse_wall_at_folds_and_gaps() {
        let la = z("America/Los_Angeles");
        // Fold: 01:30 happens at -07:00 then at -08:00; the earlier is chosen.
        assert_eq!(
            la.parse_wall(wall(2026, 11, 1, 1, 30)),
            Some(at("2026-11-01T08:30:00Z"))
        );
        // Gap: 02:30 does not exist; read with -08:00 it is 10:30 UTC = 03:30 PDT.
        let gap = la.parse_wall(wall(2026, 3, 8, 2, 30)).unwrap();
        assert_eq!(gap, at("2026-03-08T10:30:00Z"));
        assert_eq!(la.wall(gap), wall(2026, 3, 8, 3, 30));
        // Ordinary times, every kind of zone.
        assert_eq!(
            la.parse_wall(wall(2026, 11, 16, 14, 2)),
            Some(at("2026-11-16T22:02:00Z"))
        );
        assert_eq!(
            Zone::Utc.parse_wall(wall(2026, 11, 16, 14, 2)),
            Some(at("2026-11-16T14:02:00Z"))
        );
        assert_eq!(
            z("+05:30").parse_wall(wall(2026, 11, 16, 14, 2)),
            Some(at("2026-11-16T08:32:00Z"))
        );
        // Berlin: gap 02:00-03:00 on 2026-03-29, fold 02:00-03:00 on 2026-10-25.
        let berlin = z("Europe/Berlin");
        assert_eq!(
            berlin.parse_wall(wall(2026, 3, 29, 2, 15)),
            Some(at("2026-03-29T01:15:00Z"))
        );
        assert_eq!(
            berlin.parse_wall(wall(2026, 10, 25, 2, 15)),
            Some(at("2026-10-25T00:15:00Z"))
        );
        // Local: wall -> instant -> wall is the identity on an ordinary day.
        let w = wall(2026, 6, 15, 12, 0);
        assert_eq!(Zone::Local.wall(Zone::Local.parse_wall(w).unwrap()), w);
    }

    #[test]
    fn browser_names_are_safe() {
        assert_eq!(
            z("America/Los_Angeles").browser_name(),
            "America/Los_Angeles"
        );
        assert_eq!(
            z("America/Port-au-Prince").browser_name(),
            "America/Port-au-Prince"
        );
        assert_eq!(z("Etc/GMT+5").browser_name(), "Etc/GMT+5");
        assert_eq!(Zone::Utc.browser_name(), "UTC");
        assert_eq!(z("+08:00").browser_name(), "+08:00");
        assert_eq!(z("-05:30").browser_name(), "-05:30");
        let local = Zone::Local.browser_name();
        assert!(!local.is_empty());
        assert!(local.chars().all(browser_safe), "{local}");
        // An IANA name, or the offset when the computer's zone is not known.
        assert!(
            local.parse::<chrono_tz::Tz>().is_ok()
                || (local.len() == 6 && local.starts_with(['+', '-'])),
            "{local}"
        );
    }

    #[test]
    fn the_process_wide_zone_is_set_and_changed() {
        let _g = TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let before = zone();
        let t = at("2026-11-16T22:02:11.583Z");
        set(z("+08:00"));
        assert_eq!(zone(), z("+08:00"));
        assert_eq!(log_stamp(t), "2026-11-17T06:02:11.583+08:00");
        assert_eq!(display(t), "2026-11-17 06:02 +08:00");
        assert_eq!(display_date(t), "2026-11-17");
        assert_eq!(day(t), NaiveDate::from_ymd_opt(2026, 11, 17).unwrap());
        assert_eq!(name_stamp(t), "20261117-060211");
        assert_eq!(frame_stamp(t), "20261117T060211583p0800");
        assert_eq!(browser_name(), "+08:00");
        assert_eq!(
            parse_wall(wall(2026, 11, 17, 6, 2)),
            Some(at("2026-11-16T22:02:00Z"))
        );
        assert!(!change(z("+08:00")), "the same zone is no change");
        assert!(change(Zone::Utc));
        assert_eq!(zone(), Zone::Utc);
        assert_eq!(set(before), Zone::Utc);
    }
}
