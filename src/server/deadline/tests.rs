//! The write-stall and body-idle deadlines on in-memory pipes and streamed bodies (short
//! timeouts; the real-socket tests are in `listen/tests.rs`).

use super::*;
use axum::Router;
use axum::http::StatusCode;
use axum::routing::post;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;

const SHORT: Duration = Duration::from_millis(250);

/// A peer that never reads: the write fails with TimedOut after the timeout, not before.
#[tokio::test]
async fn a_write_the_peer_never_takes_times_out() {
    let (server, _client) = tokio::io::duplex(64);
    let mut guarded = WriteStall::new(server, SHORT);
    let started = Instant::now();
    let e = tokio::time::timeout(Duration::from_secs(10), guarded.write_all(&[7u8; 4096]))
        .await
        .expect("the stalled write ends")
        .unwrap_err();
    assert_eq!(e.kind(), io::ErrorKind::TimedOut);
    assert!(e.to_string().contains("stopped reading"), "{e}");
    let took = started.elapsed();
    assert!(took >= SHORT && took < Duration::from_secs(5), "{took:?}");
}

/// A reader slower than the whole transfer's length but quicker than the timeout between reads
/// is never cut: every byte arrives.
#[tokio::test]
async fn a_slow_but_moving_reader_gets_everything() {
    let (server, mut client) = tokio::io::duplex(64);
    let mut guarded = WriteStall::new(server, SHORT);
    let total = 64 * 24;
    let reader = tokio::spawn(async move {
        let mut got = 0;
        let mut buf = [0u8; 64];
        loop {
            tokio::time::sleep(Duration::from_millis(40)).await;
            match client.read(&mut buf).await.unwrap() {
                0 => return got,
                n => got += n,
            }
        }
    });
    let started = Instant::now();
    guarded.write_all(&vec![1u8; total]).await.unwrap();
    guarded.shutdown().await.unwrap();
    drop(guarded);
    assert_eq!(reader.await.unwrap(), total);
    // The transfer took several timeouts' worth: the window restarted at each read.
    assert!(started.elapsed() > SHORT * 2, "{:?}", started.elapsed());
}

/// Nothing to write (an idle keep-alive connection): no timer, flushes answer at once, and a
/// later write that the peer takes goes through.
#[tokio::test]
async fn an_idle_connection_is_not_touched() {
    let (server, mut client) = tokio::io::duplex(64);
    let mut guarded = WriteStall::new(server, SHORT);
    tokio::time::sleep(SHORT * 2).await;
    guarded.flush().await.unwrap();
    guarded.write_all(b"ok").await.unwrap();
    let mut buf = [0u8; 2];
    client.read_exact(&mut buf).await.unwrap();
    assert_eq!(&buf, b"ok");
    assert!(guarded.timer.sleep.is_none(), "no timer was ever armed");
}

/// A body fed by `rx` (each message one frame; the body ends when the sender goes).
fn streamed(rx: tokio::sync::mpsc::Receiver<&'static [u8]>) -> Body {
    Body::from_stream(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|chunk| (Ok::<_, io::Error>(Bytes::from_static(chunk)), rx))
    }))
}

/// A route that reads its whole body (as `Form` and `Json` do) under a `SHORT` idle deadline.
fn app() -> Router {
    Router::new()
        .route(
            "/up",
            post(|body: Bytes| async move { body.len().to_string() }),
        )
        .route_layer(axum::middleware::map_request(|req: Request| async move {
            with_idle_deadline(req, SHORT)
        }))
}

async fn send(body: Body) -> (StatusCode, String) {
    let req = axum::http::Request::post("/up").body(body).unwrap();
    let r = tokio::time::timeout(Duration::from_secs(10), app().oneshot(req))
        .await
        .expect("an answer")
        .unwrap();
    let status = r.status();
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// A body that stops arriving ends after the idle timeout, with a sentence saying so.
#[tokio::test]
async fn a_stalled_body_ends_after_the_idle_timeout() {
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    tx.send(&b"ab"[..]).await.unwrap();
    let started = Instant::now();
    let (status, text) = send(streamed(rx)).await;
    let took = started.elapsed();
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    assert!(
        text.contains("no part of the request body arrived"),
        "{text}"
    );
    assert!(took >= SHORT && took < Duration::from_secs(5), "{took:?}");
    drop(tx);
}

/// The control of the test above: without the idle deadline the same stalled body keeps the
/// request waiting (here four timeouts long), so that test fails without it.
#[tokio::test]
async fn without_the_deadline_a_stalled_body_waits() {
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    tx.send(&b"ab"[..]).await.unwrap();
    let bare = Router::new().route(
        "/up",
        post(|body: Bytes| async move { body.len().to_string() }),
    );
    let req = axum::http::Request::post("/up").body(streamed(rx)).unwrap();
    assert!(
        tokio::time::timeout(SHORT * 4, bare.oneshot(req))
            .await
            .is_err(),
        "a stalled body ended without the deadline"
    );
    drop(tx);
}

/// A body that trickles in, each part inside the window, completes however long it takes in
/// total (an idle deadline, not a total one).
#[tokio::test]
async fn a_trickling_body_completes() {
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    let feeder = tokio::spawn(async move {
        for _ in 0..12 {
            tokio::time::sleep(Duration::from_millis(60)).await;
            tx.send(&b"xyz"[..]).await.unwrap();
        }
    });
    let started = Instant::now();
    let (status, text) = send(streamed(rx)).await;
    assert_eq!((status, text.as_str()), (StatusCode::OK, "36"));
    assert!(started.elapsed() > SHORT * 2, "{:?}", started.elapsed());
    feeder.await.unwrap();
}

/// A body that is all there at once (the usual case) passes untouched, sizes included.
#[tokio::test]
async fn a_whole_body_passes_untouched() {
    let body = IdleBody::new(Body::from("hello"), SHORT);
    assert_eq!(hyper::body::Body::size_hint(&body).exact(), Some(5));
    assert_eq!(
        send(Body::from("hello")).await,
        (StatusCode::OK, "5".into())
    );
}
