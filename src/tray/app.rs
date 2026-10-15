//! The tray event loop (Windows and macOS): a tao event loop on the main thread owns the tray
//! icon and its menu; a poll thread queries the service manager and `/v1/status` every 3 s (and
//! at once after an action) and sends the result through an `EventLoopProxy`; start / stop /
//! restart run one at a time on a worker thread (the tray shows Busy meanwhile). The tray icon is
//! created on `StartCause::Init`, as tray-icon requires on macOS; on macOS the app has no Dock
//! icon (accessory activation policy).

use super::config::{cli_path, exe_dir, read_service_listen, service_addr, web_ui_url};
use super::elevate::{self, Outcome};
use super::icon::{ICON_SIZE, icon_rgba};
use super::model::{self, Display, IconColor, Inputs};
use super::probe::{self, HttpStatus, PROBE_TIMEOUT};
use super::{TrayArgs, single_instance};
use crate::service_ctl::{self, Action, ServiceState};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};
use tao::event::{Event, StartCause};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

/// Time between two polls.
const POLL_EVERY: Duration = Duration::from_secs(3);
/// How long a failed action's message stays in the tooltip.
const ERROR_SHOWN_FOR: Duration = Duration::from_secs(30);

/// One poll result.
#[derive(Debug)]
struct Poll {
    service: Option<ServiceState>,
    http: HttpStatus,
}

#[derive(Debug)]
enum UserEvent {
    Menu(MenuEvent),
    Status(Poll),
    ActionDone(Action, Outcome),
}

/// The tray icon and the menu items it updates.
struct Ui {
    tray: TrayIcon,
    status: MenuItem,
    open_ui: MenuItem,
    start: MenuItem,
    stop: MenuItem,
    restart: MenuItem,
    quit: MenuItem,
    shown: Option<Display>,
}

impl Ui {
    fn build(d: &Display) -> anyhow::Result<Self> {
        let status = MenuItem::new(&d.status_line, false, None);
        let open_ui = MenuItem::new("Open Web UI", d.can_open_ui, None);
        let start = MenuItem::new("Start Service", d.can_start, None);
        let stop = MenuItem::new("Stop Service", d.can_stop, None);
        let restart = MenuItem::new("Restart Service", d.can_restart, None);
        let quit = MenuItem::new("Quit Tray (the service keeps running)", true, None);
        let menu = Menu::new();
        menu.append_items(&[
            &status,
            &PredefinedMenuItem::separator(),
            &open_ui,
            &PredefinedMenuItem::separator(),
            &start,
            &stop,
            &restart,
            &PredefinedMenuItem::separator(),
            &quit,
        ])?;
        let tray = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_icon(icon(d.color)?)
            .with_tooltip(&d.tooltip)
            .with_menu_on_left_click(true)
            .build()?;
        Ok(Self {
            tray,
            status,
            open_ui,
            start,
            stop,
            restart,
            quit,
            shown: Some(d.clone()),
        })
    }

    fn show(&mut self, d: &Display) {
        if self.shown.as_ref() == Some(d) {
            return;
        }
        let old_color = self.shown.as_ref().map(|s| s.color);
        if self.shown.as_ref().map(|s| &s.status_line) != Some(&d.status_line) {
            tracing::info!("{}", d.status_line);
        }
        self.status.set_text(&d.status_line);
        self.open_ui.set_enabled(d.can_open_ui);
        self.start.set_enabled(d.can_start);
        self.stop.set_enabled(d.can_stop);
        self.restart.set_enabled(d.can_restart);
        if let Err(e) = self.tray.set_tooltip(Some(&d.tooltip)) {
            tracing::warn!("could not set the tray tooltip: {e}");
        }
        if old_color != Some(d.color) {
            match icon(d.color) {
                Ok(i) => {
                    if let Err(e) = self.tray.set_icon(Some(i)) {
                        tracing::warn!("could not set the tray icon: {e}");
                    }
                }
                Err(e) => tracing::warn!("{e:#}"),
            }
        }
        self.shown = Some(d.clone());
    }
}

fn icon(color: IconColor) -> anyhow::Result<Icon> {
    Ok(Icon::from_rgba(
        icon_rgba(color, ICON_SIZE),
        ICON_SIZE,
        ICON_SIZE,
    )?)
}

/// Everything the event loop tracks besides the widgets.
struct State {
    addr: SocketAddr,
    cli: PathBuf,
    last: Option<Poll>,
    busy: Option<Action>,
    error: Option<(String, Instant)>,
    poke: mpsc::Sender<()>,
    proxy: EventLoopProxy<UserEvent>,
}

impl State {
    fn display(&mut self) -> Display {
        if self
            .error
            .as_ref()
            .is_some_and(|(_, at)| at.elapsed() > ERROR_SHOWN_FOR)
        {
            self.error = None;
        }
        model::display(&Inputs {
            service: self.last.as_ref().and_then(|p| p.service),
            http: self.last.as_ref().map(|p| &p.http),
            busy: self.busy,
            error: self.error.as_ref().map(|(m, _)| m.as_str()),
        })
    }

    fn start_action(&mut self, action: Action) {
        if self.busy.is_some() {
            return;
        }
        self.busy = Some(action);
        self.error = None;
        let cli = self.cli.clone();
        let proxy = self.proxy.clone();
        tracing::info!("{} the service through {}", action.verb(), cli.display());
        std::thread::spawn(move || {
            let outcome = elevate::run_elevated(&cli, action);
            let _ = proxy.send_event(UserEvent::ActionDone(action, outcome));
        });
    }

    fn finish_action(&mut self, action: Action, outcome: Outcome) {
        self.busy = None;
        match outcome {
            Outcome::Done => tracing::info!("service {} done", action.verb()),
            Outcome::Cancelled => {
                tracing::info!("service {}: cancelled by the user", action.verb())
            }
            Outcome::Failed(m) => {
                tracing::warn!("service {} failed: {m}", action.verb());
                elevate::show_error(&m);
                self.error = Some((m, Instant::now()));
            }
        }
        let _ = self.poke.send(());
    }
}

/// `service_ctl::query()`, `None` when it fails (or panics: the poll thread must survive).
fn query_service() -> Option<ServiceState> {
    match std::panic::catch_unwind(service_ctl::query) {
        Ok(Ok(s)) => Some(s),
        Ok(Err(e)) => {
            tracing::debug!("service query failed: {e:#}");
            None
        }
        Err(_) => {
            tracing::warn!("service query panicked");
            None
        }
    }
}

fn poll_loop(addr: SocketAddr, proxy: EventLoopProxy<UserEvent>, poke: mpsc::Receiver<()>) {
    loop {
        let poll = Poll {
            service: query_service(),
            http: probe::probe(addr, PROBE_TIMEOUT),
        };
        if proxy.send_event(UserEvent::Status(poll)).is_err() {
            return;
        }
        match poke.recv_timeout(POLL_EVERY) {
            Ok(()) | Err(RecvTimeoutError::Timeout) => while poke.try_recv().is_ok() {},
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn init_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .try_init();
}

pub(super) fn run(args: TrayArgs) -> ExitCode {
    init_logging();
    let _lock = match single_instance() {
        Ok(Some(f)) => Some(f),
        Ok(None) => {
            tracing::info!("the tray already runs for this user");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            tracing::warn!("single-instance lock unavailable ({e}); continuing");
            None
        }
    };
    let addr = service_addr(args.port, read_service_listen());
    let cli = cli_path(args.cli.as_deref(), exe_dir().as_deref());
    tracing::info!(
        "service at http://{addr}/, controlled through {}",
        cli.display()
    );

    #[allow(unused_mut, reason = "mutated on macOS only")]
    let mut event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    #[cfg(target_os = "macos")]
    {
        use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS as _};
        event_loop.set_activation_policy(ActivationPolicy::Accessory);
    }
    let proxy = event_loop.create_proxy();
    let menu_proxy = proxy.clone();
    MenuEvent::set_event_handler(Some(move |e| {
        let _ = menu_proxy.send_event(UserEvent::Menu(e));
    }));
    let (poke_tx, poke_rx) = mpsc::channel();
    let poll_proxy = proxy.clone();
    if let Err(e) = std::thread::Builder::new()
        .name("tray-poll".into())
        .spawn(move || poll_loop(addr, poll_proxy, poke_rx))
    {
        tracing::error!("could not start the poll thread: {e}");
        return ExitCode::FAILURE;
    }

    let mut state = State {
        addr,
        cli,
        last: None,
        busy: None,
        error: None,
        poke: poke_tx,
        proxy,
    };
    let mut ui: Option<Ui> = None;
    event_loop.run(move |event, _, flow| {
        *flow = ControlFlow::Wait;
        match event {
            Event::NewEvents(StartCause::Init) => match Ui::build(&state.display()) {
                Ok(u) => ui = Some(u),
                Err(e) => {
                    tracing::error!("could not create the tray icon: {e:#}");
                    *flow = ControlFlow::Exit;
                    return;
                }
            },
            Event::UserEvent(UserEvent::Status(p)) => state.last = Some(p),
            Event::UserEvent(UserEvent::ActionDone(a, o)) => state.finish_action(a, o),
            Event::UserEvent(UserEvent::Menu(e)) => {
                let Some(u) = ui.as_ref() else { return };
                if e.id == *u.quit.id() {
                    ui = None; // removes the icon
                    *flow = ControlFlow::Exit;
                    return;
                } else if e.id == *u.open_ui.id() {
                    elevate::open_url(&web_ui_url(state.addr));
                } else if e.id == *u.start.id() {
                    state.start_action(Action::Start);
                } else if e.id == *u.stop.id() {
                    state.start_action(Action::Stop);
                } else if e.id == *u.restart.id() {
                    state.start_action(Action::Restart);
                }
            }
            _ => return,
        }
        let d = state.display();
        if let Some(u) = ui.as_mut() {
            u.show(&d);
        }
    })
}
