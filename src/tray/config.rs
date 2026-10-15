//! Where the tray finds the service: the HTTP port and address (from `--port`, else the
//! installed service's config when the user can read it, else 32168) and the CLI used to control
//! the service (`--cli`, else the installed location). The path lookups are thin; the parsing
//! and address math are pure and unit-tested.

use serde_json::Value;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};

/// Port of the service when nothing says otherwise (CodeProject.AI's).
pub const DEFAULT_PORT: u16 = 32168;

/// Port and listen address read from a service config.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServiceListen {
    pub port: Option<u16>,
    pub bind_address: Option<IpAddr>,
}

/// The installed service's config file.
pub fn service_config_path() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        let root = std::env::var_os("ProgramData")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"));
        Some(
            root.join("BlueOnyxPrism")
                .join("blue_onyx_prism_config_service.json"),
        )
    }
    #[cfg(target_os = "macos")]
    {
        Some(PathBuf::from(
            "/usr/local/var/blue-onyx-prism/blue_onyx_prism_config.json",
        ))
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        None
    }
}

/// Read the service config; `None` when missing or unreadable (macOS: mode 0640, so usually
/// not readable by the logged-in user).
pub fn read_service_listen() -> ServiceListen {
    service_config_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| parse_service_listen(&t))
        .unwrap_or_default()
}

/// `port` and `bind_address` of a config JSON (missing or invalid fields are `None`).
pub fn parse_service_listen(text: &str) -> ServiceListen {
    let Ok(v) = serde_json::from_str::<Value>(text) else {
        return ServiceListen::default();
    };
    ServiceListen {
        port: v
            .get("port")
            .and_then(Value::as_u64)
            .and_then(|p| u16::try_from(p).ok())
            .filter(|p| *p != 0),
        bind_address: v
            .get("bind_address")
            .and_then(Value::as_str)
            .and_then(|s| s.parse().ok()),
    }
}

/// The address to probe and open in the browser: loopback when the service listens on every
/// interface (or loopback), else the address it binds.
pub fn service_addr(arg_port: Option<u16>, listen: ServiceListen) -> SocketAddr {
    let port = arg_port.or(listen.port).unwrap_or(DEFAULT_PORT);
    let ip = match listen.bind_address {
        None => IpAddr::V4(Ipv4Addr::LOCALHOST),
        Some(IpAddr::V4(a)) if a.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        Some(IpAddr::V6(a)) if a.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        Some(a) => a,
    };
    SocketAddr::new(ip, port)
}

/// `http://<addr>/` (IPv6 in brackets).
pub fn web_ui_url(addr: SocketAddr) -> String {
    format!("http://{addr}/")
}

/// File name of the CLI binary.
pub const CLI_NAME: &str = if cfg!(windows) {
    "blue-onyx-prism.exe"
} else {
    "blue-onyx-prism"
};

/// Where the macOS package installs the CLI.
pub const MACOS_CLI: &str = "/usr/local/blue-onyx-prism/blue-onyx-prism";

/// The CLI used to control the service: `--cli`, else on macOS the installed CLI, else the CLI
/// next to the tray executable (`exe_dir`).
pub fn cli_path(arg: Option<&Path>, exe_dir: Option<&Path>) -> PathBuf {
    if let Some(p) = arg {
        return p.to_path_buf();
    }
    if cfg!(target_os = "macos") {
        return PathBuf::from(MACOS_CLI);
    }
    match exe_dir {
        Some(d) => d.join(CLI_NAME),
        None => PathBuf::from(CLI_NAME),
    }
}

/// Directory of the running executable.
pub fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_port_and_bind_address() {
        let l = parse_service_listen(r#"{"port": 8080, "bind_address": "0.0.0.0", "x": 1}"#);
        assert_eq!(l.port, Some(8080));
        assert_eq!(l.bind_address, Some(IpAddr::V4(Ipv4Addr::UNSPECIFIED)));
        assert_eq!(parse_service_listen("not json"), ServiceListen::default());
        assert_eq!(parse_service_listen(r#"{"port": 70000}"#).port, None);
        assert_eq!(parse_service_listen(r#"{"port": 0}"#).port, None);
        assert_eq!(parse_service_listen(r#"{"port": "80"}"#).port, None);
    }

    #[test]
    fn port_precedence() {
        let cfg = ServiceListen {
            port: Some(9000),
            bind_address: None,
        };
        assert_eq!(service_addr(Some(1234), cfg).port(), 1234);
        assert_eq!(service_addr(None, cfg).port(), 9000);
        assert_eq!(
            service_addr(None, ServiceListen::default()),
            "127.0.0.1:32168".parse().unwrap()
        );
    }

    #[test]
    fn address_follows_the_bind_address() {
        let at = |b: &str| {
            service_addr(
                None,
                ServiceListen {
                    port: Some(1),
                    bind_address: Some(b.parse().unwrap()),
                },
            )
        };
        assert_eq!(at("0.0.0.0"), "127.0.0.1:1".parse().unwrap());
        assert_eq!(at("::"), "[::1]:1".parse().unwrap());
        assert_eq!(at("127.0.0.1"), "127.0.0.1:1".parse().unwrap());
        assert_eq!(at("192.168.1.5"), "192.168.1.5:1".parse().unwrap());
        assert_eq!(web_ui_url(at("::")), "http://[::1]:1/");
        assert_eq!(web_ui_url(at("0.0.0.0")), "http://127.0.0.1:1/");
    }

    #[test]
    fn cli_path_precedence() {
        let arg = Path::new("/opt/x/blue-onyx-prism");
        assert_eq!(cli_path(Some(arg), None), arg);
        let got = cli_path(None, Some(Path::new("/a/b")));
        if cfg!(target_os = "macos") {
            assert_eq!(got, Path::new(MACOS_CLI));
        } else {
            assert_eq!(got, Path::new("/a/b").join(CLI_NAME));
        }
    }
}
