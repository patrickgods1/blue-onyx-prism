//! What an import reads: local video files and image folders (Blue Iris clip and alert folders
//! included), a video URL, or pinned open footage. Pure planning plus a directory walk:
//! [`discover`] lists the videos and images under a path (Blue Iris `.bvr` clips are counted and
//! explained, never read), [`check_url`] refuses URLs of unsecured-camera aggregators, internal
//! hosts and anything but http(s), [`check_resolved`] refuses names that resolve to an internal
//! address, [`check_license`] enforces the per-import licence confirmation, and
//! [`blue_iris_time`] / [`camera_of`] read Blue Iris file names (`Driveway.20240315_083012.mp4`).

use anyhow::{Result, bail};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use tracing::debug;

/// Video extensions read by the import (PyAV / FFmpeg decodes them).
pub const VIDEO_EXTS: [&str; 4] = ["mp4", "mkv", "avi", "mov"];
/// Image extensions read by the import.
pub const IMAGE_EXTS: [&str; 4] = ["jpg", "jpeg", "png", "bmp"];
/// Most files a folder import lists.
pub const MAX_FILES: usize = 20_000;
/// Deepest folder level walked below the import path.
pub const MAX_DEPTH: usize = 6;

/// Hosts (and their subdomains) that list unsecured or private cameras without their owners'
/// consent. Footage from them is never imported, whatever licence is claimed.
pub const DENYLIST: [&str; 12] = [
    "insecam.org",
    "insecam.com",
    "insecam.net",
    "insecam.cc",
    "opentopia.com",
    "shodan.io",
    "camhacker.com",
    "camhackers.com",
    "lookcam.com",
    "unsecuredcams.com",
    "unsecuredcameras.com",
    "openipcam.com",
];

/// Where the frames come from.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase", tag = "kind", content = "value")]
pub enum ImportSource {
    /// A video file, an image, or a folder of them (a Blue Iris clip / alert folder).
    Path(PathBuf),
    /// A video page or file URL, downloaded with yt-dlp.
    Url(String),
    /// Pinned open footage (`meva-sample`).
    Dataset(String),
}

impl ImportSource {
    /// Short text for titles and the provenance (`clip.mp4`, `https://...`, `meva-sample`).
    pub fn describe(&self) -> String {
        match self {
            ImportSource::Path(p) => p.file_name().map_or_else(
                || p.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            ),
            ImportSource::Url(u) => u.clone(),
            ImportSource::Dataset(d) => d.clone(),
        }
    }
}

fn ext_of(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default()
}

/// A video file the import reads.
pub fn is_video(p: &Path) -> bool {
    VIDEO_EXTS.contains(&ext_of(p).as_str())
}

/// An image file the import reads.
pub fn is_image(p: &Path) -> bool {
    IMAGE_EXTS.contains(&ext_of(p).as_str())
}

/// Why a listing stopped before the end of the walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Stop {
    /// [`MAX_FILES`] videos and images listed, with more entries left.
    Files,
    /// [`Budget::max_dirs`] folders read.
    Folders,
    /// [`Budget::deadline`] passed.
    Time,
    /// [`Budget::cancel`] said so (a newer preview).
    Cancelled,
}

/// Limits of a walk on top of [`MAX_FILES`] and [`MAX_DEPTH`] (the preview's budget; an import
/// lists with [`Budget::default`], no further limit).
#[derive(Default)]
pub struct Budget<'a> {
    /// Most videos and images listed (None = [`MAX_FILES`]; smaller in tests).
    pub max_files: Option<usize>,
    /// Most folders read (`read_dir` calls).
    pub max_dirs: Option<usize>,
    /// The walk stops when this passes (checked per folder and every [`CHECK_EVERY`] entries).
    pub deadline: Option<std::time::Instant>,
    /// Stop when this answers true (checked like the deadline).
    pub cancel: Option<&'a (dyn Fn() -> bool + Sync)>,
    /// Sum the sizes of the videos and images into [`Discovered::bytes`].
    pub sizes: bool,
}

/// Entries between two checks of the deadline and the cancel flag inside one folder.
pub const CHECK_EVERY: usize = 256;

impl std::fmt::Debug for Budget<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Budget")
            .field("max_files", &self.max_files)
            .field("max_dirs", &self.max_dirs)
            .field("deadline", &self.deadline)
            .field("cancel", &self.cancel.is_some())
            .field("sizes", &self.sizes)
            .finish()
    }
}

impl Budget<'_> {
    fn interrupted(&self) -> Option<Stop> {
        if self.cancel.is_some_and(|c| c()) {
            return Some(Stop::Cancelled);
        }
        if self
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            return Some(Stop::Time);
        }
        None
    }
}

/// What a path holds.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct Discovered {
    pub videos: Vec<PathBuf>,
    pub images: Vec<PathBuf>,
    /// Blue Iris `.bvr` clips (not readable: export them to MP4 in Blue Iris).
    pub bvr: usize,
    /// Other files skipped.
    pub other: usize,
    /// The listing stopped early ([`Discovered::stopped`] says why).
    pub truncated: bool,
    pub stopped: Option<Stop>,
    /// Folders read.
    pub dirs: usize,
    /// Size of the videos and images (only with [`Budget::sizes`]).
    pub bytes: u64,
}

impl Discovered {
    /// Warnings for the report.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.bvr > 0 {
            w.push(format!(
                "{} Blue Iris .bvr clip(s) skipped: .bvr is Blue Iris's own format and cannot be \
                 read; export the clips to MP4 in Blue Iris (select the clips, right-click, Export, \
                 format MP4) and import the exported files",
                self.bvr
            ));
        }
        match self.stopped {
            Some(Stop::Files) => w.push(format!(
                "only the first {MAX_FILES} videos and images are listed; import sub-folders one \
                 at a time"
            )),
            Some(Stop::Folders | Stop::Time) => w.push(format!(
                "a large folder: the preview counted the first {} folders only; an import lists \
                 up to {MAX_FILES} videos and images",
                self.dirs
            )),
            Some(Stop::Cancelled) | None => {}
        }
        w
    }
}

/// The videos and images at `path` (a file, or a folder walked [`MAX_DEPTH`] levels deep; hidden
/// entries and symlinked folders skipped), sorted by path. The walk stops as soon as
/// [`MAX_FILES`] videos and images are listed.
pub fn discover(path: &Path) -> Result<Discovered> {
    discover_with(path, &Budget::default())
}

/// [`discover`] within `budget`: the walk stops (`truncated`, [`Discovered::stopped`]) at the
/// first limit reached and never reads another folder after it. A cancelled walk answers what
/// it found (maybe nothing) without an error; a walk stopped by the budget before finding
/// anything is an error that says so.
pub fn discover_with(path: &Path, budget: &Budget<'_>) -> Result<Discovered> {
    let meta = std::fs::metadata(path)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
    let mut d = Discovered::default();
    if meta.is_file() {
        classify(path, &mut d);
        if budget.sizes && (!d.videos.is_empty() || !d.images.is_empty()) {
            d.bytes = meta.len();
        }
        if d.videos.is_empty() && d.images.is_empty() {
            if d.bvr > 0 {
                bail!("{}", d.warnings().join("; "));
            }
            bail!(
                "{} is not a video ({}) or image ({}) file",
                path.display(),
                VIDEO_EXTS.join(", "),
                IMAGE_EXTS.join(", ")
            );
        }
        return Ok(d);
    }
    d.stopped = walk(path, budget, &mut d);
    d.truncated = d.stopped.is_some();
    d.videos.sort();
    d.images.sort();
    let empty = d.videos.is_empty() && d.images.is_empty();
    if d.stopped == Some(Stop::Cancelled) {
        return Ok(d);
    }
    if empty && matches!(d.stopped, Some(Stop::Folders | Stop::Time)) {
        bail!(
            "{} is a large folder: no video ({}) or image ({}) files in the first {} folders the \
             preview reads; choose the sub-folder that holds the clips",
            path.display(),
            VIDEO_EXTS.join(", "),
            IMAGE_EXTS.join(", "),
            d.dirs
        );
    }
    if empty {
        let mut why = format!(
            "{} has no video ({}) or image ({}) files",
            path.display(),
            VIDEO_EXTS.join(", "),
            IMAGE_EXTS.join(", ")
        );
        for w in d.warnings() {
            why.push_str("; ");
            why.push_str(&w);
        }
        bail!("{why}");
    }
    Ok(d)
}

/// The folder walk of [`discover_with`]: why it stopped early, None when it listed everything.
fn walk(path: &Path, budget: &Budget<'_>, d: &mut Discovered) -> Option<Stop> {
    let mut stack = vec![(path.to_path_buf(), 0usize)];
    let mut seen = 0usize;
    let mut tick = |budget: &Budget<'_>| -> Option<Stop> {
        seen += 1;
        if seen.is_multiple_of(CHECK_EVERY) {
            budget.interrupted()
        } else {
            None
        }
    };
    while let Some((dir, depth)) = stack.pop() {
        if let Some(stop) = budget.interrupted() {
            return Some(stop);
        }
        if budget.max_dirs.is_some_and(|m| d.dirs >= m) {
            return Some(Stop::Folders);
        }
        d.dirs += 1;
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut entries = Vec::new();
        for e in rd.flatten() {
            entries.push(e);
            if let Some(stop) = tick(budget) {
                return Some(stop);
            }
        }
        entries.sort_by_key(std::fs::DirEntry::file_name);
        let mut rest = entries.into_iter().peekable();
        while let Some(e) = rest.next() {
            if let Some(stop) = tick(budget) {
                return Some(stop);
            }
            if e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                if depth < MAX_DEPTH {
                    stack.push((e.path(), depth + 1));
                }
            } else if ft.is_file() {
                let p = e.path();
                let media = is_video(&p) || is_image(&p);
                classify(&p, d);
                if media && budget.sizes {
                    d.bytes += e.metadata().map_or(0, |m| m.len());
                }
                if media && d.videos.len() + d.images.len() >= budget.max_files.unwrap_or(MAX_FILES)
                {
                    // Full: stop here; "more" when anything is left to look at.
                    let more = rest.peek().is_some() || !stack.is_empty();
                    return more.then_some(Stop::Files);
                }
            }
        }
    }
    None
}

fn classify(p: &Path, d: &mut Discovered) {
    if is_video(p) {
        d.videos.push(p.to_path_buf());
    } else if is_image(p) {
        d.images.push(p.to_path_buf());
    } else if ext_of(p) == "bvr" {
        d.bvr += 1;
    } else {
        d.other += 1;
    }
}

/// The authority of a lowercase http(s) URL: what lies between `//` and the path.
fn authority_of(url: &str) -> Option<&str> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    rest.split(['/', '?', '#']).next()
}

/// The host of an http(s) URL, lowercase, without credentials, port or a trailing dot.
fn host_of(url: &str) -> Option<String> {
    let host_port = authority_of(url)?.rsplit('@').next()?;
    let host = if let Some(v6) = host_port.strip_prefix('[') {
        v6.split(']').next()?.to_string()
    } else {
        host_port.split(':').next()?.to_string()
    };
    let h = host.trim_end_matches('.').to_ascii_lowercase();
    (!h.is_empty()).then_some(h)
}

/// Name endings that only ever mean this machine or the local network, whatever a resolver
/// answers.
const INTERNAL_SUFFIXES: [&str; 3] = [".localhost", ".local", ".internal"];

/// The next action in every "this host is internal" refusal.
const INTERNAL_HINT: &str = "a URL import only fetches public sites, because the job runs with \
                             the service's network access. For a video on your own network, \
                             copy it to this machine and import the file";

/// One part of a number-form IPv4 address: decimal, octal after a `0`, hex after `0x`. Lowercase
/// input (see [`host_of`]).
fn ipv4_part(part: &str) -> Option<u64> {
    let (digits, radix) = match (part.strip_prefix("0x"), part.strip_prefix('0')) {
        (Some(hex), _) => (hex, 16),
        (None, Some(oct)) if !oct.is_empty() => (oct, 8),
        _ => (part, 10),
    };
    if digits.is_empty() {
        // `0x` alone: 0 for URL parsers, so it must not pass as a name.
        return (radix == 16).then_some(0);
    }
    // `from_str_radix` also takes a sign, which no address form has.
    if !digits.chars().all(|c| c.is_digit(radix)) {
        return None;
    }
    u64::from_str_radix(digits, radix).ok()
}

/// `host` read the way `inet_aton` reads it, as `getaddrinfo`, curl and browsers do: one to four
/// parts, each decimal, octal or hex, the last filling every byte left (`127.1`, `2130706433`,
/// `0x7f.0.0.1`, `0177.0.0.1`). `None` when it is not one.
fn parse_ipv4(host: &str) -> Option<Ipv4Addr> {
    let parts: Vec<&str> = host.split('.').collect();
    let (last, head) = parts.split_last()?;
    let largest_last: u32 = match head.len() {
        0 => u32::MAX,
        1 => 0x00ff_ffff,
        2 => 0xffff,
        3 => 0xff,
        _ => return None,
    };
    let mut bytes = [0u8; 4];
    for (slot, part) in bytes.iter_mut().zip(head) {
        *slot = u8::try_from(ipv4_part(part)?).ok()?;
    }
    let last = u32::try_from(ipv4_part(last)?).ok()?;
    if last > largest_last {
        return None;
    }
    let value = u32::from_be_bytes(bytes) | last;
    Some(Ipv4Addr::from(value))
}

/// The last label of `host` is a number, so resolvers and URL parsers read the whole host as an
/// address, never as a name: no top-level domain is numeric.
fn ends_in_number(host: &str) -> bool {
    let last = host.rsplit('.').next().unwrap_or(host);
    match last.strip_prefix("0x") {
        Some(hex) => hex.chars().all(|c| c.is_ascii_hexdigit()),
        None => !last.is_empty() && last.chars().all(|c| c.is_ascii_digit()),
    }
}

/// An IPv4 address that is not on the public internet: loopback, private, link-local, CGNAT,
/// "this network" (0.0.0.0/8), multicast, reserved or broadcast.
fn is_internal_v4(ip: Ipv4Addr) -> bool {
    let [a, b, ..] = ip.octets();
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || a == 0
        || (a == 100 && (64..128).contains(&b))
        || ip.is_multicast()
        || a >= 240
}

/// The IPv4 address that traffic for `ip` ends up at, for the forms that carry one: IPv4-mapped
/// and IPv4-translated (`::ffff:0:0/96` around them), IPv4-compatible, NAT64 (`64:ff9b::/96`)
/// and 6to4 (`2002::/16`).
fn embedded_v4(ip: Ipv6Addr) -> Option<Ipv4Addr> {
    let [.., o12, o13, o14, o15] = ip.octets();
    let [_, _, o2, o3, o4, o5, ..] = ip.octets();
    match ip.segments() {
        [0, 0, 0, 0, 0, 0xffff, ..]
        | [0, 0, 0, 0, 0xffff, 0, ..]
        | [0, 0, 0, 0, 0, 0, ..]
        | [0x64, 0xff9b, 0, 0, 0, 0, ..] => Some(Ipv4Addr::new(o12, o13, o14, o15)),
        [0x2002, ..] => Some(Ipv4Addr::new(o2, o3, o4, o5)),
        _ => None,
    }
}

/// An IPv6 address that is not on the public internet: unspecified, loopback, link-local,
/// site-local, unique-local, multicast, or a form around an internal IPv4 address.
fn is_internal_v6(ip: Ipv6Addr) -> bool {
    let [first, ..] = ip.segments();
    ip.is_unspecified()
        || ip.is_loopback()
        || (first & 0xffc0) == 0xfe80
        || (first & 0xffc0) == 0xfec0
        || (first & 0xfe00) == 0xfc00
        || ip.is_multicast()
        || embedded_v4(ip).is_some_and(is_internal_v4)
}

/// `ip` is not on the public internet.
fn is_internal_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_internal_v4(v4),
        IpAddr::V6(v6) => is_internal_v6(v6),
    }
}

/// `host` (from [`host_of`]) as an address literal in any form it can be spelled.
fn literal_ip(host: &str) -> Option<IpAddr> {
    host.parse::<Ipv6Addr>()
        .ok()
        .map(IpAddr::V6)
        .or_else(|| parse_ipv4(host).map(IpAddr::V4))
}

/// Refuse a URL whose host is internal, as far as the text tells: an address in any spelling, or
/// a local name. Anything a URL parser might read differently from [`host_of`] is refused too,
/// since the host checked here has to be the host yt-dlp connects to.
fn check_host(url: &str, host: &str) -> Result<()> {
    let authority = authority_of(url).unwrap_or_default();
    // A backslash ends the authority for urllib3 but not for `urlsplit`, and yt-dlp holds both.
    if authority.contains('\\') {
        bail!("the URL is not valid: its host has a backslash; write it with forward slashes");
    }
    if authority
        .rsplit('@')
        .next()
        .is_some_and(|h| h.starts_with('['))
    {
        return match host.parse::<Ipv6Addr>() {
            Ok(ip) if is_internal_v6(ip) => {
                bail!("{host} is a private or local address: {INTERNAL_HINT}")
            }
            Ok(_) => Ok(()),
            Err(_) => bail!("the URL is not valid: {host} is not an IPv6 address"),
        };
    }
    // Only plain ASCII names. A resolver folds Unicode (ideographic dots, full-width digits,
    // circled letters) to ASCII and `%` escapes are decoded by some parsers, so either could
    // spell `127.0.0.1` past the checks below. An international name has an `xn--` form.
    if !host
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_'))
    {
        bail!(
            "the URL is not valid: its host {host:?} has characters a host name cannot hold; \
             write an international name in its xn-- form"
        );
    }
    if ends_in_number(host) {
        let Some(ip) = parse_ipv4(host) else {
            bail!(
                "the URL is not valid: {host} is no IP address, and a host name cannot end in a \
                 number"
            );
        };
        if is_internal_v4(ip) {
            bail!("{host} is a private or local address ({ip}): {INTERNAL_HINT}");
        }
        return Ok(());
    }
    if host == "localhost" || INTERNAL_SUFFIXES.iter().any(|s| host.ends_with(s)) {
        bail!("{host} is a private or local name: {INTERNAL_HINT}");
    }
    Ok(())
}

/// Check a video URL: http(s) only,/// Check a video URL: http(s) only, a host, not on the [`DENYLIST`] of unsecured-camera
/// aggregators (the host or any parent domain), and not an internal host (see [`check_host`]).
/// Returns the URL trimmed. Pure: no network, so it also runs when a queued job starts.
///
/// This stops a URL from pointing at the service's own network; [`check_resolved`] does the same
/// for a name that resolves there. Neither covers redirects: yt-dlp follows them in Python, so a
/// public URL that redirects to a private address still gets there, and so does a name whose
/// DNS answer changes after the plan. Closing that needs a hook inside yt-dlp.
pub fn check_url(url: &str) -> Result<String> {
    let u = url.trim();
    if u.is_empty() {
        bail!("no URL given");
    }
    if u.chars().any(char::is_control) || u.len() > 2048 {
        bail!("the URL is not valid");
    }
    let lower = u.to_ascii_lowercase();
    if !(lower.starts_with("https://") || lower.starts_with("http://")) {
        bail!(
            "only http:// and https:// video URLs can be imported (got {}); for a camera stream, \
             record a clip and import the file",
            u.split(':').next().unwrap_or("?")
        );
    }
    let Some(host) = host_of(&lower) else {
        bail!("the URL has no host");
    };
    if let Some(bad) = DENYLIST
        .iter()
        .find(|d| host == **d || host.ends_with(&format!(".{d}")))
    {
        bail!(
            "{host} is refused: {bad} lists unsecured or private cameras whose owners did not \
             agree to be watched; footage from such sites is never imported. Use your own \
             cameras' clips or footage published under a licence that allows it"
        );
    }
    check_host(&lower, &host)?;
    Ok(u.to_string())
}

/// Addresses of `host` from the system resolver.
pub fn system_resolve(host: &str) -> std::io::Result<Vec<IpAddr>> {
    Ok((host, 0u16).to_socket_addrs()?.map(|a| a.ip()).collect())
}

/// The second half of [`check_url`], at plan time: resolve the URL's host once through `resolve`
/// and refuse it when any answer is internal. A resolver failure carries on: yt-dlp reports a
/// name that does not resolve in its own words, and an offline planner must not guess. An
/// address literal needs no lookup. See [`check_url`] for what this does not cover.
pub fn check_resolved(
    url: &str,
    resolve: &dyn Fn(&str) -> std::io::Result<Vec<IpAddr>>,
) -> Result<()> {
    let Some(host) = host_of(url.to_ascii_lowercase().as_str()) else {
        return Ok(());
    };
    if literal_ip(&host).is_some() {
        return Ok(());
    }
    let answers = match resolve(&host) {
        Ok(a) => a,
        Err(e) => {
            debug!("resolving {host} for the import failed, yt-dlp will report it: {e}");
            return Ok(());
        }
    };
    if let Some(ip) = answers.into_iter().find(|ip| is_internal_ip(*ip)) {
        bail!("{host} resolves to a private or local address ({ip}): {INTERNAL_HINT}");
    }
    Ok(())
}

/// The licence of an import: the stated one (or `default` when allowed) and the confirmation,
/// which a file or URL import must carry. Errors say what to pass.
pub fn check_license(
    source: &ImportSource,
    license: Option<&str>,
    confirmed: bool,
) -> Result<String> {
    let stated = license.map(str::trim).filter(|l| !l.is_empty());
    match source {
        ImportSource::Dataset(_) => Ok(stated.unwrap_or_default().to_string()),
        ImportSource::Url(_) => {
            let Some(l) = stated else {
                bail!(
                    "state the video's licence (--license, e.g. \"CC BY 4.0\" or \"own footage\") \
                     and confirm you may use it for training (--confirm-license)"
                );
            };
            if !confirmed {
                bail!(
                    "confirm that the licence \"{l}\" allows you to use this video for training \
                     (--confirm-license, or tick the box on the Import tab)"
                );
            }
            if l.chars().count() > 200 || l.chars().any(char::is_control) {
                bail!("the licence text must be one line of at most 200 characters");
            }
            Ok(l.to_string())
        }
        ImportSource::Path(_) => {
            let l = stated.unwrap_or("own footage");
            if !confirmed {
                bail!(
                    "confirm that you may use this footage for training (licence \"{l}\"; \
                     --confirm-license, or tick the box on the Import tab)"
                );
            }
            if l.chars().count() > 200 || l.chars().any(char::is_control) {
                bail!("the licence text must be one line of at most 200 characters");
            }
            Ok(l.to_string())
        }
    }
}

/// The recording time of a Blue Iris file name (`Driveway.20240315_083012.mp4`,
/// `Cam1.20240315_083012.123456.3-1.jpg`): the first `.YYYYMMDD_HHMMSS` part, as local time
/// is unknown taken as UTC.
pub fn blue_iris_time(name: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
    for part in file.split('.').skip(1) {
        let b = part.as_bytes();
        if b.len() >= 15
            && b.get(8) == Some(&b'_')
            && b.get(..8).is_some_and(|d| d.iter().all(u8::is_ascii_digit))
            && b.get(9..15)
                .is_some_and(|d| d.iter().all(u8::is_ascii_digit))
        {
            let s = part.get(..15)?;
            if let Ok(t) = chrono::NaiveDateTime::parse_from_str(s, "%Y%m%d_%H%M%S") {
                return Some(t.and_utc());
            }
        }
    }
    None
}

/// The part of a Blue Iris recording exported in segments (`Back.20261010_160000_3.mp4`: every
/// part carries the recording's start time, numbered `_1`, `_2`, ...): the input up to and
/// including the time (folder, camera and time: the recording's key) and the part number; `None`
/// for any other name.
pub fn blue_iris_segment(name: &str) -> Option<(&str, u32)> {
    let start = name.rfind(['/', '\\']).map_or(0, |i| i + 1);
    let file = name.get(start..)?;
    let mut at = file.find('.')? + 1;
    for part in file.split('.').skip(1) {
        let b = part.as_bytes();
        let digits = |r: std::ops::Range<usize>| {
            b.get(r)
                .is_some_and(|d| !d.is_empty() && d.iter().all(u8::is_ascii_digit))
        };
        if b.len() > 16
            && b.len() <= 21
            && digits(0..8)
            && b.get(8) == Some(&b'_')
            && digits(9..15)
            && b.get(15) == Some(&b'_')
            && digits(16..b.len())
        {
            let n = part.get(16..)?.parse().ok()?;
            return Some((name.get(..start + at + 15)?, n));
        }
        at += part.len() + 1;
    }
    None
}

/// Start offsets (seconds after the time in the name) of `videos` (path, length in seconds): the
/// parts of one segmented Blue Iris recording ([`blue_iris_segment`], same folder, camera and
/// time) follow each other in part order, each after the lengths of the lower parts in the list
/// (a part of unknown length counts as 0); every other video starts at 0. Without this, all the
/// parts of a recording claim the same minute and the dedup drops them as near duplicates.
pub fn segment_offsets(videos: &[(&str, Option<f64>)]) -> Vec<f64> {
    let mut by_rec: std::collections::BTreeMap<&str, Vec<(u32, usize)>> =
        std::collections::BTreeMap::new();
    for (i, (name, _)) in videos.iter().enumerate() {
        if let Some((key, n)) = blue_iris_segment(name) {
            by_rec.entry(key).or_default().push((n, i));
        }
    }
    let mut out = vec![0.0; videos.len()];
    for parts in by_rec.values_mut() {
        parts.sort_unstable();
        let mut at = 0.0;
        for (_, i) in parts.iter() {
            if let Some(o) = out.get_mut(*i) {
                *o = at;
            }
            let len = videos.get(*i).and_then(|v| v.1);
            at += len.filter(|d| d.is_finite() && *d > 0.0).unwrap_or(0.0);
        }
    }
    out
}

/// The recording time of a MEVA clip name (`2018-03-11.23-55-00.00-00-00.school.G639.r13.avi`).
pub fn meva_time(name: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let mut parts = file.split('.');
    let (d, t) = (parts.next()?, parts.next()?);
    chrono::NaiveDateTime::parse_from_str(&format!("{d} {t}"), "%Y-%m-%d %H-%M-%S")
        .ok()
        .map(|x| x.and_utc())
}

/// The camera folder of a file: `camera` when given, else the file name's camera part the way
/// the collector names Blue Iris uploads (`Driveway.20240315_083012.mp4` -> `driveway`; a MEVA
/// clip -> `<site>-<camera>`), else `import`.
pub fn camera_of(file: &Path, camera: Option<&str>) -> String {
    let fallback = |s: &str| crate::collect::layout::camera_dir(&format!("import|cam:{s}"), 0, 0);
    if let Some(c) = camera.map(str::trim).filter(|c| !c.is_empty()) {
        return fallback(c);
    }
    let name = file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if meva_time(&name).is_some() {
        let parts: Vec<&str> = name.split('.').collect();
        if let (Some(site), Some(cam)) = (parts.get(3), parts.get(4)) {
            return fallback(&format!("meva-{site}-{cam}"));
        }
    }
    let key = crate::static_memory::camera_key("import", &name, 0, 0);
    if key.contains("|cam:") {
        crate::collect::layout::camera_dir(&key, 0, 0)
    } else {
        "import".into()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Every spelling of an internal address is refused without touching the network.
    #[test]
    fn internal_hosts_are_refused() {
        for bad in [
            // Loopback, private, link-local, CGNAT, unspecified.
            "http://127.0.0.1/v.mp4",
            "http://127.1.2.3:8080/v.mp4",
            "http://10.1.2.3/v.mp4",
            "http://172.16.0.1/v.mp4",
            "http://172.31.255.255/v.mp4",
            "http://192.168.1.10/v.mp4",
            "http://169.254.169.254/latest/meta-data/",
            "http://100.64.0.1/v.mp4",
            "http://100.127.255.255/v.mp4",
            "http://0.0.0.0/v.mp4",
            "http://0.1.2.3/v.mp4",
            "http://224.0.0.1/v.mp4",
            "http://255.255.255.255/v.mp4",
            "http://240.0.0.1/v.mp4",
            // Userinfo tricks: the host is what follows the last '@'.
            "http://example.com@127.0.0.1/",
            "http://user:pw@192.168.0.1:8080/x",
            "http://a@b@10.0.0.1/",
            "http://example.com:80@127.0.0.1/",
            // Number forms `inet_aton` accepts.
            "http://2130706433/",
            "http://017700000001/",
            "http://0x7f000001/",
            "http://0X7F000001/",
            "http://0x7f.0.0.1/",
            "http://0177.0.0.1/",
            "http://0177.0.0.0x1/",
            "http://127.1/",
            "http://127.0.1/",
            "http://0x7f.1/",
            "http://10.0xffffff/",
            "http://192.168.257/",
            "http://0/",
            "http://0x/",
            "http://2852039166/",
            // A trailing dot changes nothing.
            "http://127.0.0.1./",
            "http://2130706433./",
            // IPv6: loopback, unspecified, link-local, unique-local, site-local, multicast.
            "http://[::1]/v.mp4",
            "http://[::1]:8080/v.mp4",
            "http://[0:0:0:0:0:0:0:1]/",
            "http://[::]/",
            "http://[fe80::1]/",
            "http://[febf::1]/",
            "http://[fc00::1]/",
            "http://[fd12:3456::1]/",
            "http://[fec0::1]/",
            "http://[ff02::1]/",
            // IPv4 inside IPv6: mapped, compatible, NAT64, 6to4.
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:7f00:1]/",
            "http://[::ffff:192.168.1.1]/",
            "http://[::ffff:0:a00:1]/",
            "http://[::127.0.0.1]/",
            "http://[64:ff9b::7f00:1]/",
            "http://[2002:7f00:1::]/",
            "http://[2002:c0a8:101::1]/",
            // Names.
            "http://localhost/",
            "http://LOCALHOST:8080/",
            "http://localhost./",
            "http://foo.localhost/",
            "http://nas.local/",
            "http://NAS.LOCAL./",
            "http://metadata.google.internal/computeMetadata/v1/",
        ] {
            let e = check_url(bad).unwrap_err().to_string();
            assert!(
                e.contains("private or local"),
                "{bad}: the wrong refusal: {e}"
            );
            assert!(e.contains("import the file"), "{bad}: {e}");
        }
    }

    /// Hosts no resolver should see: a parser that reads them differently from this one could
    /// reach a host this check never saw.
    #[test]
    fn hosts_that_parsers_disagree_on_are_refused() {
        for bad in [
            // A backslash ends the authority for urllib3, not for `urlsplit`.
            "http://127.0.0.1\\@example.com/",
            "http://example.com\\@127.0.0.1/",
            "http://example.com\\.127.0.0.1/",
            // Numbers out of range: not an address, and no real name ends in a number.
            "http://127.0.0.256/",
            "http://1.2.3.4.5/",
            "http://4294967296/",
            "http://6442450945/",
            "http://0x17f000001/",
            "http://0189.0.0.1/",
            "http://256.1.1.1/",
            "http://127..0.1/",
            "http://example.123/",
            // Percent escapes, zone ids and bad brackets.
            "http://%31%32%37.0.0.1/",
            "http://[fe80::1%25eth0]/",
            "http://[not-an-address]/",
            "http://[::1/",
            // Unicode a resolver folds to ASCII: ideographic dots, full-width digits, circled
            // letters.
            "http://127\u{3002}0\u{3002}0\u{3002}1/",
            "http://\u{ff11}\u{ff12}\u{ff17}.0.0.1/",
            "http://\u{24db}\u{24de}\u{24d2}\u{24d0}\u{24db}\u{24d7}\u{24de}\u{24e2}\u{24e3}/",
            "http://b\u{fc}cher.example/",
            // A space in the host.
            "http://exa mple.com/",
        ] {
            assert!(check_url(bad).is_err(), "{bad}");
        }
    }

    /// Public hosts keep working: a public address, and names that merely look internal.
    #[test]
    fn public_hosts_pass_the_internal_check() {
        for ok in [
            "https://www.youtube.com/watch?v=abc",
            "http://8.8.8.8/v.mp4",
            "http://[2001:4860:4860::8888]/v.mp4",
            "http://[::ffff:8.8.8.8]/v.mp4",
            "http://[2002:808:808::1]/v.mp4",
            "http://172.15.255.255/",
            "http://172.32.0.1/",
            "http://100.63.255.255/",
            "http://100.128.0.1/",
            "http://192.167.1.1/",
            "http://169.253.1.1/",
            "http://134744072/",
            "http://0x08080808/",
            "https://mylocalhost.com/v",
            "https://localhost.example.com/v",
            "https://local.example.com/v",
            "https://example.internal.com/v",
            "https://notlocal/v",
            "https://cdn-1.example.com/v.mp4",
            "https://1.example.com/v.mp4",
            "https://xn--bcher-kva.example/v.mp4",
            "https://example.com:8443/v.mp4",
            "https://user:pw@example.com/v.mp4",
            "https://example.com/127.0.0.1/\\x",
            "https://example.com/?u=http://127.0.0.1/",
        ] {
            assert!(check_url(ok).is_ok(), "{ok}: {:?}", check_url(ok));
        }
    }

    /// The number forms read as `inet_aton` reads them.
    #[test]
    fn ipv4_number_forms() {
        let v4 = |a, b, c, d| Some(std::net::Ipv4Addr::new(a, b, c, d));
        assert_eq!(parse_ipv4("127.0.0.1"), v4(127, 0, 0, 1));
        assert_eq!(parse_ipv4("0177.0.0.01"), v4(127, 0, 0, 1));
        assert_eq!(parse_ipv4("0x7f.0x0.0x0.0x1"), v4(127, 0, 0, 1));
        assert_eq!(parse_ipv4("127.1"), v4(127, 0, 0, 1));
        assert_eq!(parse_ipv4("127.0.1"), v4(127, 0, 0, 1));
        assert_eq!(parse_ipv4("1.2.772"), v4(1, 2, 3, 4));
        assert_eq!(parse_ipv4("1.131844"), v4(1, 2, 3, 4));
        assert_eq!(parse_ipv4("2130706433"), v4(127, 0, 0, 1));
        assert_eq!(parse_ipv4("037777777777"), v4(255, 255, 255, 255));
        assert_eq!(parse_ipv4("0xffffffff"), v4(255, 255, 255, 255));
        assert_eq!(parse_ipv4("0"), v4(0, 0, 0, 0));
        assert_eq!(
            parse_ipv4("00000000000000000000000000177.1"),
            v4(127, 0, 0, 1)
        );
        for bad in [
            "",
            ".",
            "1.2.3.256",
            "1.2.65536",
            "1.16777216",
            "4294967296",
            "256.1",
            "1.2.3.4.5",
            "08",
            "0x1g",
            "+1",
            "1.+2",
            "a.b",
        ] {
            assert_eq!(parse_ipv4(bad), None, "{bad}");
        }
    }

    /// A fake resolver with fixed answers.
    fn answers(ips: &'static [&'static str]) -> impl Fn(&str) -> std::io::Result<Vec<IpAddr>> {
        move |_| Ok(ips.iter().map(|s| s.parse().unwrap()).collect())
    }

    /// One private answer among public ones is enough to refuse; the fake sees the bare host.
    #[test]
    fn a_name_resolving_to_a_private_address_is_refused() {
        let url = "https://user@Sneaky.Example.com.:8443/v.mp4";
        for ips in [
            &["192.168.1.1"][..],
            &["8.8.8.8", "127.0.0.1"],
            &["2001:4860:4860::8888", "::1"],
            &["::ffff:10.0.0.1"],
        ] {
            let seen = std::cell::RefCell::new(Vec::new());
            let fake = |h: &str| -> std::io::Result<Vec<IpAddr>> {
                seen.borrow_mut().push(h.to_string());
                Ok(ips.iter().map(|s| s.parse().unwrap()).collect())
            };
            let e = check_resolved(url, &fake).unwrap_err().to_string();
            assert!(e.contains("sneaky.example.com"), "{e}");
            assert!(e.contains("private or local"), "{e}");
            assert!(e.contains("import the file"), "{e}");
            assert_eq!(*seen.borrow(), ["sneaky.example.com"], "resolved once");
        }
    }

    /// Public answers pass; a resolver failure or no answer carries on (yt-dlp reports it).
    #[test]
    fn public_answers_and_resolver_failures_carry_on() {
        let url = "https://example.com/v.mp4";
        check_resolved(url, &answers(&["93.184.216.34", "2606:2800:220:1::1"])).unwrap();
        check_resolved(url, &answers(&[])).unwrap();
        let broken =
            |_: &str| -> std::io::Result<Vec<IpAddr>> { Err(std::io::Error::other("no DNS")) };
        check_resolved(url, &broken).unwrap();
    }

    /// The real resolver, on the one name that needs no network.
    #[test]
    fn the_system_resolver_answers_for_localhost() {
        let ips = system_resolve("localhost").unwrap();
        assert!(!ips.is_empty());
        assert!(ips.iter().all(|ip| ip.is_loopback()), "{ips:?}");
        let e = check_resolved("http://localhost/", &system_resolve).unwrap_err();
        assert!(e.to_string().contains("resolves to"), "{e}");
    }

    /// An address literal needs no lookup: `check_url` has read it already.
    #[test]
    fn address_literals_are_not_resolved() {
        let never = |h: &str| -> std::io::Result<Vec<IpAddr>> { panic!("resolved {h}") };
        check_resolved("http://8.8.8.8/v.mp4", &never).unwrap();
        check_resolved("http://134744072/v.mp4", &never).unwrap();
        check_resolved("http://[2001:4860:4860::8888]/v.mp4", &never).unwrap();
    }

    #[test]
    fn denylisted_hosts_and_other_schemes_are_refused() {
        for bad in [
            "http://www.insecam.org/en/view/123/",
            "https://INSECAM.com/cam",
            "https://images.shodan.io/x.jpg",
            "http://user:pw@opentopia.com:8080/webcam",
            "https://insecam.org./x",
        ] {
            let e = check_url(bad).unwrap_err().to_string();
            assert!(e.contains("is refused"), "{bad}: {e}");
            assert!(e.contains("unsecured"), "{bad}: {e}");
        }
        for bad in [
            "rtsp://192.168.1.10/stream",
            "file:///etc/passwd",
            "ftp://x/y.mp4",
            "",
            "https://",
        ] {
            assert!(check_url(bad).is_err(), "{bad}");
        }
        assert_eq!(
            check_url(" https://www.youtube.com/watch?v=abc ").unwrap(),
            "https://www.youtube.com/watch?v=abc"
        );
        // A host that only contains a denied name is fine.
        assert!(check_url("https://notinsecam.org.example.com/v.mp4").is_ok());
        assert!(check_url("https://example.com/insecam.org.mp4").is_ok());
        assert_eq!(
            host_of("https://[::1]:8080/x").as_deref(),
            Some("::1"),
            "IPv6 literal"
        );
    }

    #[test]
    fn licence_confirmation_is_required_for_files_and_urls() {
        let url = ImportSource::Url("https://example.com/v.mp4".into());
        let path = ImportSource::Path("clip.mp4".into());
        let ds = ImportSource::Dataset("meva-sample".into());
        assert!(
            check_license(&url, None, true)
                .unwrap_err()
                .to_string()
                .contains("--license")
        );
        assert!(
            check_license(&url, Some("CC BY 4.0"), false)
                .unwrap_err()
                .to_string()
                .contains("--confirm-license")
        );
        assert_eq!(
            check_license(&url, Some(" CC BY 4.0 "), true).unwrap(),
            "CC BY 4.0"
        );
        assert!(check_license(&path, None, false).is_err());
        assert_eq!(check_license(&path, None, true).unwrap(), "own footage");
        assert_eq!(check_license(&ds, None, false).unwrap(), "");
        assert!(check_license(&url, Some("a\nb"), true).is_err());
    }

    #[test]
    fn blue_iris_and_meva_names() {
        let t = blue_iris_time("Driveway.20240315_083012.mp4").unwrap();
        assert_eq!(t.to_rfc3339(), "2024-03-15T08:30:12+00:00");
        assert!(
            blue_iris_time("D:\\BlueIris\\Alerts\\Cam1.20240315_083012.123456.3-1.jpg").is_some()
        );
        assert!(
            blue_iris_time("Cam1.20241315_083012.mp4").is_none(),
            "month 13"
        );
        assert!(blue_iris_time("clip.mp4").is_none());
        let seg = "/v/Export/Back.20261010_160000_12.mp4";
        assert!(blue_iris_time(seg).is_some());
        assert_eq!(
            blue_iris_segment(seg),
            Some(("/v/Export/Back.20261010_160000", 12))
        );
        assert_eq!(
            blue_iris_segment("D:\\Clips\\Front Door.20261010_160000_1.mp4"),
            Some(("D:\\Clips\\Front Door.20261010_160000", 1))
        );
        for other in [
            "Driveway.20240315_083012.mp4",
            "Cam1.20240315_083012.123456.3-1.jpg",
            "Back.20261010_160000_.mp4",
            "Back.20261010_160000_x1.mp4",
            "Back.20261010_160000_123456.mp4",
            "20261010_160000_3.mp4",
            "clip.mp4",
        ] {
            assert_eq!(blue_iris_segment(other), None, "{other}");
        }
        let m = meva_time("2018-03-11.23-55-00.00-00-00.school.G639.r13.avi").unwrap();
        assert_eq!(m.to_rfc3339(), "2018-03-11T23:55:00+00:00");
        assert_eq!(
            camera_of(Path::new("/x/Driveway.20240315_083012.mp4"), None),
            "driveway"
        );
        assert_eq!(
            camera_of(
                Path::new("2018-03-11.23-55-00.00-00-00.school.G639.r13.avi"),
                None
            ),
            "meva-school-g639"
        );
        assert_eq!(camera_of(Path::new("image.jpg"), None), "import");
        assert_eq!(
            camera_of(Path::new("a.mp4"), Some("Back Yard")),
            "back_yard"
        );
    }

    #[test]
    fn discover_lists_videos_images_and_explains_bvr() {
        let dir = std::env::temp_dir().join(format!("bop-import-src-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("Clips/sub")).unwrap();
        std::fs::create_dir_all(dir.join(".hidden")).unwrap();
        for f in [
            "Clips/Drive.20240101_010101.mp4",
            "Clips/Drive.20240101_020202.bvr",
            "Clips/sub/b.MKV",
            "Alerts.jpg",
            "notes.txt",
            ".hidden/x.mp4",
        ] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let d = discover(&dir).unwrap();
        assert_eq!(d.videos.len(), 2);
        assert_eq!(d.images, [dir.join("Alerts.jpg")]);
        assert_eq!((d.bvr, d.other), (1, 1));
        assert!(d.warnings()[0].contains("export the clips to MP4"));
        let one = discover(&dir.join("Clips/sub/b.MKV")).unwrap();
        assert_eq!(one.videos.len(), 1);
        let e = discover(&dir.join("Clips/Drive.20240101_020202.bvr")).unwrap_err();
        assert!(e.to_string().contains(".bvr"), "{e}");
        assert!(discover(&dir.join("notes.txt")).is_err());
        assert!(discover(&dir.join("missing")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_budget_stops_the_walk_without_reading_further_folders() {
        let dir = std::env::temp_dir().join(format!("bop-import-budget-{}", uuid::Uuid::new_v4()));
        // 30 folders of one clip each, plus a notes file.
        for i in 0..30 {
            let sub = dir.join(format!("d{i:02}"));
            std::fs::create_dir_all(&sub).unwrap();
            std::fs::write(sub.join("Cam.20240101_010101.mp4"), b"vid").unwrap();
        }
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        let all = discover_with(
            &dir,
            &Budget {
                sizes: true,
                ..Budget::default()
            },
        )
        .unwrap();
        assert_eq!((all.videos.len(), all.dirs, all.truncated), (30, 31, false));
        assert_eq!(all.stopped, None);
        assert_eq!(all.bytes, 90);
        // At most 5 folders read: the root and 4 sub-folders.
        let d = discover_with(
            &dir,
            &Budget {
                max_dirs: Some(5),
                sizes: true,
                ..Budget::default()
            },
        )
        .unwrap();
        assert_eq!((d.videos.len(), d.dirs), (4, 5));
        assert_eq!(d.stopped, Some(Stop::Folders));
        assert!(d.truncated);
        assert_eq!(d.bytes, 12);
        assert!(
            d.warnings()[0].contains("first 5 folders"),
            "{:?}",
            d.warnings()
        );
        // A deadline in the past reads nothing; with nothing found that is an error that says
        // the folder is large.
        let e = discover_with(
            &dir,
            &Budget {
                deadline: Some(std::time::Instant::now()),
                ..Budget::default()
            },
        )
        .unwrap_err();
        assert!(e.to_string().contains("is a large folder"), "{e}");
        // A cancelled walk answers what it has, without an error.
        let yes = || true;
        let c = discover_with(
            &dir,
            &Budget {
                cancel: Some(&yes),
                ..Budget::default()
            },
        )
        .unwrap();
        assert_eq!(
            (c.stopped, c.dirs, c.videos.len()),
            (Some(Stop::Cancelled), 0, 0)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_file_cap_stops_at_once_and_only_claims_more_when_there_is_more() {
        let dir = std::env::temp_dir().join(format!("bop-import-cap-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("a")).unwrap();
        for i in 0..10 {
            std::fs::write(dir.join("a").join(format!("{i:02}.jpg")), b"").unwrap();
        }
        let cap = |n| Budget {
            max_files: Some(n),
            ..Budget::default()
        };
        // Exactly the cap: not truncated.
        let d = discover_with(&dir, &cap(10)).unwrap();
        assert_eq!((d.images.len(), d.truncated), (10, false));
        // Entries left in the folder: more.
        let d = discover_with(&dir, &cap(4)).unwrap();
        assert_eq!((d.images.len(), d.stopped), (4, Some(Stop::Files)));
        // Folders are read last-sorted first: `a`, then `0extra`, which is never read once `a`
        // fills the cap (the walk stops at once), yet makes it "more".
        std::fs::create_dir_all(dir.join("0extra/c")).unwrap();
        std::fs::write(dir.join("0extra/c/x.mp4"), b"").unwrap();
        let d = discover_with(&dir, &cap(10)).unwrap();
        assert_eq!((d.images.len(), d.videos.len()), (10, 0));
        assert_eq!(d.stopped, Some(Stop::Files));
        assert_eq!(d.dirs, 2, "the root and `a` only");
        assert!(
            d.warnings()[0].contains("only the first 20000"),
            "{:?}",
            d.warnings()
        );
        // Without the cap everything is listed.
        let d = discover(&dir).unwrap();
        assert_eq!((d.images.len(), d.videos.len(), d.dirs), (10, 1, 4));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn segments_of_one_recording_follow_each_other() {
        // Out of order, a gap (no part 3), another recording, another folder, a plain clip and a
        // part of unknown length.
        let v = [
            ("/e/Back.20261010_160000_2.mp4", Some(63.3)),
            ("/e/Back.20261010_160000_1.mp4", Some(35.9)),
            ("/e/Back.20261010_160000_4.mp4", Some(64.0)),
            ("/e/Back.20261010_170042_1.mp4", Some(60.0)),
            ("/e/Back.20261010_170042_2.mp4", None),
            ("/e/Back.20261010_170042_3.mp4", Some(60.0)),
            ("/f/Back.20261010_160000_2.mp4", Some(10.0)),
            ("/e/Driveway.20261010_160000.mp4", Some(300.0)),
        ];
        let o = segment_offsets(&v);
        let want = [35.9, 0.0, 35.9 + 63.3, 0.0, 60.0, 60.0, 0.0, 0.0];
        assert_eq!(o.len(), want.len());
        for (i, (a, b)) in o.iter().zip(want).enumerate() {
            assert!((a - b).abs() < 1e-9, "{i}: {a} != {b}");
        }
        assert!(segment_offsets(&[]).is_empty());
    }
}
