//! Deadlines for a connection's peer once its request head is in (audit 2026-11, RA-032): hyper
//! 1.x times only the request head and the keep-alive wait ([`super::listen::HEADER_READ_TIMEOUT`]),
//! so without these a client that stops reading its answer, or stops sending a request body,
//! held its connection, its task and its place under the connection caps for ever (a few LAN
//! hosts could fill [`super::listen::MAX_CONNECTIONS`] and lock Blue Iris out).
//!
//! - [`WriteStall`] wraps every accepted socket, below TLS (the handshake, the answers and the
//!   TLS alerts are covered): a write or flush that has waited on the peer for
//!   [`WRITE_STALL_TIMEOUT`] fails with [`std::io::ErrorKind::TimedOut`], hyper closes the
//!   connection and its slot under the caps goes back. A reader that takes any bytes at all
//!   within the window restarts it: a slow but moving client is never cut, and an idle keep-alive
//!   connection (nothing to write) is not touched.
//! - [`IdleBody`] wraps the request body of the signed-in and sign-in routes ([`idle_body`],
//!   attached in [`super::router`]): when no part of the body has arrived for
//!   [`BODY_IDLE_TIMEOUT`] while the handler waits for it, the body ends with an error (axum
//!   answers 400, or 500 for a multipart upload, and hyper closes the connection). It is an idle
//!   deadline, not a total one, so a large upload of any length works as long as bytes keep
//!   coming; nothing is buffered here, so the per-route `DefaultBodyLimit`s still apply. The
//!   detection routes keep their own total read deadline (`detect`, 10 s) and do not get it.
//!
//! Both arm a timer only while the peer is the one being waited for (the inner poll answered
//! `Pending`), so a handler that takes long to read its body or to produce its answer never
//! counts against the peer, and an active connection costs no timer work.

use axum::BoxError;
use axum::body::{Body, Bytes};
use axum::extract::Request;
use hyper::body::{Frame, SizeHint};
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::time::Sleep;

/// Longest a write to a connection may wait on its peer to take bytes before the connection is
/// closed. Long enough for a phone whose browser is briefly busy and for TCP's retransmission
/// backoff after lost packets (1 + 2 + 4 + 8 s); Blue Iris reads its small answers at once; short
/// enough that a client that never reads gives its slot back within half a minute. The wait is
/// what the server sees, not the client's reads: the kernel reports a full socket writable again
/// only after a large part of its send buffer has drained (some 340 KB on macOS loopback, a third
/// of the autotuned buffer on Linux), so a client reading slower than about 20 KB/s for a long
/// answer could be cut; no LAN client that reads at all is that slow.
pub(crate) const WRITE_STALL_TIMEOUT: Duration = Duration::from_secs(30);

/// Longest the signed-in routes wait for the next part of a request body before ending it. An
/// upload whose network stops for this long has failed already (TCP gives up on a retransmission
/// much later, but a browser shows the stall); a large upload that keeps sending is never cut.
/// Three times the detection route's whole-upload deadline, which suits single frames only.
pub(crate) const BODY_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

/// A timer armed only while the peer is being waited for: [`Self::poll`] when the inner call
/// answered `Pending`, [`Self::clear`] when it made progress. The [`Sleep`] is allocated at the
/// first stall and reused after it.
#[derive(Debug)]
struct StallTimer {
    timeout: Duration,
    sleep: Option<Pin<Box<Sleep>>>,
    armed: bool,
}

impl StallTimer {
    fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            sleep: None,
            armed: false,
        }
    }

    /// The peer made progress: the next stall starts a new window.
    fn clear(&mut self) {
        self.armed = false;
    }

    /// The inner call is waiting on the peer: arm the timer if it is not armed, and whether the
    /// window has run out (the caller fails the call). Registers `cx` to wake at the deadline.
    fn expired(&mut self, cx: &mut Context<'_>) -> bool {
        if !self.armed {
            let deadline = tokio::time::Instant::now() + self.timeout;
            match self.sleep.as_mut() {
                Some(s) => s.as_mut().reset(deadline),
                None => self.sleep = Some(Box::pin(tokio::time::sleep_until(deadline))),
            }
            self.armed = true;
        }
        let ran_out = self
            .sleep
            .as_mut()
            .is_some_and(|s| s.as_mut().poll(cx).is_ready());
        if ran_out {
            self.armed = false;
        }
        ran_out
    }

    /// `polled` with a stall past the timeout turned into `TimedOut`.
    fn guard<T>(
        &mut self,
        cx: &mut Context<'_>,
        polled: Poll<io::Result<T>>,
    ) -> Poll<io::Result<T>> {
        match polled {
            Poll::Ready(r) => {
                self.clear();
                Poll::Ready(r)
            }
            Poll::Pending if self.expired(cx) => Poll::Ready(Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "the client took no data for {} s while an answer was being sent; the \
                     connection was closed (a client that stopped reading)",
                    self.timeout.as_secs_f32()
                ),
            ))),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// An accepted socket whose writes fail after waiting on the peer for longer than the timeout
/// (see the module docs). Reads pass through untouched (hyper times the request head itself).
#[derive(Debug)]
pub(crate) struct WriteStall<S> {
    inner: S,
    timer: StallTimer,
}

impl<S> WriteStall<S> {
    pub(crate) fn new(inner: S, timeout: Duration) -> Self {
        Self {
            inner,
            timer: StallTimer::new(timeout),
        }
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for WriteStall<S> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for WriteStall<S> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.inner).poll_write(cx, buf);
        this.timer.guard(cx, polled)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.inner).poll_write_vectored(cx, bufs);
        this.timer.guard(cx, polled)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.inner).poll_flush(cx);
        this.timer.guard(cx, polled)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.inner).poll_shutdown(cx);
        this.timer.guard(cx, polled)
    }
}

/// The error a stalled request body ends with.
#[derive(Debug)]
pub(crate) struct BodyStalled(Duration);

impl std::fmt::Display for BodyStalled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "no part of the request body arrived for {} s (the client or its network stopped \
             sending); the request was ended, send it again",
            self.0.as_secs_f32()
        )
    }
}

impl std::error::Error for BodyStalled {}

/// A request body that ends with [`BodyStalled`] when no frame arrives within its timeout while
/// it is being read (see the module docs).
#[derive(Debug)]
pub(crate) struct IdleBody<B> {
    inner: B,
    timer: StallTimer,
}

impl<B> IdleBody<B> {
    pub(crate) fn new(inner: B, timeout: Duration) -> Self {
        Self {
            inner,
            timer: StallTimer::new(timeout),
        }
    }
}

impl<B> hyper::body::Body for IdleBody<B>
where
    B: hyper::body::Body<Data = Bytes> + Unpin,
    B::Error: Into<BoxError>,
{
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, BoxError>>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(frame) => {
                this.timer.clear();
                Poll::Ready(frame.map(|f| f.map_err(Into::into)))
            }
            Poll::Pending if this.timer.expired(cx) => {
                Poll::Ready(Some(Err(Box::new(BodyStalled(this.timer.timeout)))))
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

/// `req` with its body under an idle deadline of `timeout` (nothing is read here).
pub(crate) fn with_idle_deadline(req: Request, timeout: Duration) -> Request {
    req.map(|body| Body::new(IdleBody::new(body, timeout)))
}

/// The router's middleware ([`axum::middleware::map_request`]): the body under
/// [`BODY_IDLE_TIMEOUT`].
pub(crate) async fn idle_body(req: Request) -> Request {
    with_idle_deadline(req, BODY_IDLE_TIMEOUT)
}

#[cfg(test)]
mod tests;
