#!/usr/bin/env python
"""Footage import helpers of Blue Onyx Prism (`blue-onyx-prism train import`).

The service plans an import and writes the frames into the dataset folder itself; this script only
does what needs the training environment's libraries:

  extract    Decode a video with PyAV (its bundled FFmpeg, LGPL) and keep the frames FFmpeg's
             `select` filter passes (--select: the expression the service built from the scene-change
             threshold, the maximum frame rate and the keep-one-every interval), at most
             --max-frames. Writes f_<n>.jpg (quality 92) into --out and one `frame` event per frame
             (file, time in seconds from the first frame, stored size w/h, source size sw/sh, the
             frame's own time `pts` in seconds and its decode index `n`, and `full` when the file
             is the full-resolution frame). --max-side N stores the frames at most N px on the long
             side (the candidate pass: the analysis needs no more; the frames kept are extracted
             again at full resolution by `frames`); 0 keeps the full resolution. --hwaccel auto decodes on the platform's hardware
             decoder when this FFmpeg has one (d3d11va on Windows, cuda then vaapi on Linux; software
             on macOS, where VideoToolbox measured slower), with PyAV's software fallback for
             streams it cannot take. A decoder that cannot open the video makes it try the next
             one in that order, then software; a hardware decode error decodes the video again in
             software from the start (no seek; the frames already passed are skipped by time, or
             by count when frames carry no time). The `start` event names the decoder
             (`hw:<type>` or `sw`), a fallback sends a `decoder` event. `done` reports `capped`
             when the video had more than --max-frames to give (a further frame passed the
             filter within LOOKAHEAD_S after the quota, or decoding stopped there with frames
             left).
  frames     Extract the frames listed in --list (JSON: [{"name", "pts", "n"}], from extract's
             `frame` events) again at full resolution (quality 92, like extract without
             --max-side) into --out: a seek per frame (decoding on when the next one is close),
             matched by `pts`, else by the decode index `n` (one pass from the start). One `frame`
             event per written file (name, pts, n, w, h), a `missing` event for each frame not
             found (the file is not written), then `done`. --hwaccel as for extract.
  fetch-url  Download one video with yt-dlp (no playlists, no live streams, video-only formats up to
             --max-height so no ffmpeg binary is needed to merge streams, at most --max-bytes).
             Prints a `fetched` event with the file and what the site reports (title, uploader,
             licence, page URL) for the provenance record. The service checked the URL against its
             denylist and the user confirmed the licence before this runs.
  embed      SigLIP 2 image embeddings (L2-normalized float32, row-major) of the images listed in
             --list, written to --out; the service picks the frames (k-center greedy) from them.

Progress: one stdout line `@import {json}` per event; everything else is log output. Files are
written atomically (a temporary name, then renamed). A full disk ends the run with
"error: no space left on device ..." (the service turns it into its own message).

    python scripts/import_footage.py --self-check
    python scripts/import_footage.py --self-test     # the pure helpers, no PyAV needed
    python scripts/import_footage.py extract --video clip.mp4 --out frames/ --hwaccel auto \\
        --select "if(isnan(prev_selected_t),1,gt(scene,0.3)*gte(t-prev_selected_t,0.5)+gte(t-prev_selected_t,10))"
    python scripts/import_footage.py frames --video clip.mp4 --out full/ --list keep.json
    python scripts/import_footage.py fetch-url --url https://example.org/v.mp4 --out dl/
    python scripts/import_footage.py embed --list frames.txt --model <siglip2 dir> --out emb.f32
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[import_footage] {msg}", flush=True)


def event(kind: str, **fields) -> None:
    fields["event"] = kind
    print("\n@import " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


def fail(msg: str) -> None:
    raise SystemExit(f"[import_footage] error: {msg}")


# Errors of a full disk: ENOSPC / EDQUOT, and Windows' ERROR_DISK_FULL / ERROR_HANDLE_DISK_FULL.
DISK_FULL_ERRNOS = {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}
DISK_FULL_WINERRORS = {112, 39}


def is_disk_full(e: BaseException) -> bool:
    """Whether `e` (or what caused it) says the disk is full."""
    seen = 0
    while e is not None and seen < 8:
        if isinstance(e, OSError) and (e.errno in DISK_FULL_ERRNOS
                                       or getattr(e, "winerror", None) in DISK_FULL_WINERRORS):
            return True
        e = e.__cause__ or e.__context__
        seen += 1
    return False


def save_jpeg(img, path: Path) -> None:
    """Write `img` (PIL) as a quality-92 JPEG at `path` atomically (temporary name, rename): a
    file in --out is always complete. A full disk ends the run with a message the service
    recognizes."""
    tmp = path.with_name(f".{path.name}.part")
    try:
        img.save(tmp, format="JPEG", quality=92)
        os.replace(tmp, path)
    except OSError as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        if is_disk_full(e):
            fail(f"no space left on device writing {path} ({e})")
        raise


def scaled_size(w: int, h: int, max_side: int) -> tuple[int, int]:
    """(w, h) scaled to at most `max_side` on the long side (aspect kept, at least 1 px);
    unchanged when it already fits or `max_side` is 0."""
    if max_side <= 0 or max(w, h) <= max_side:
        return w, h
    if w >= h:
        return max_side, max(1, round(h * max_side / w))
    return max(1, round(w * max_side / h)), max_side


def watch_parent() -> None:
    if os.name == "nt":
        return
    ppid = os.getppid()
    if ppid <= 1:
        return
    import threading

    def loop() -> None:
        while True:
            time.sleep(2)
            if os.getppid() != ppid:
                log("the parent process is gone; stopping")
                os._exit(3)

    threading.Thread(target=loop, name="parent-watch", daemon=True).start()


def self_check() -> int:
    import importlib

    info = {}
    for mod in ("numpy", "PIL", "av", "yt_dlp"):
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", None) or getattr(getattr(m, "version", None), "__version__", "?")
        info[mod] = str(v)
        log(f"self-check: {mod} {v}")
    import av

    lic = {}
    meta = getattr(getattr(av, "_core", None), "library_meta", None) or {}
    for k, v in meta.items():
        lic[k] = v.get("license")
    hw = hw_devices()
    order = hw_candidates("auto")
    log(f"self-check: hardware decoders {', '.join(hw) or 'none'}; auto tries "
        f"{', '.join(order + ['software']) if order else 'software only'}")
    event("self_check", ffmpeg_license=lic.get("libavcodec"), hwdevices=hw, **info)
    return 0


# Hardware decoders tried by --hwaccel auto, in order, per platform.
# macOS is left out: through PyAV, VideoToolbox decoded 1080p clips 2-4x slower (wall) than software
# on an M1 (docs/TRAINING.md), so `auto` decodes in software there.
HW_ORDER = {"win32": ["d3d11va"], "linux": ["cuda", "vaapi"]}


def hw_devices() -> list[str]:
    """The hardware device types this PyAV's FFmpeg was built with."""
    try:
        from av.codec.hwaccel import hwdevices_available
    except ImportError:
        return []
    try:
        return list(hwdevices_available())
    except Exception:  # noqa: BLE001
        return []


def hw_candidates(choice: str) -> list[str]:
    """The hardware device types to try in order (software after them): none for `off`, the
    named one when this FFmpeg has it, or the platform's HW_ORDER entries it has for `auto`."""
    if choice == "off":
        return []
    have = hw_devices()
    if choice != "auto":
        return [choice] if choice in have else []
    plat = "linux" if sys.platform.startswith("linux") else sys.platform
    return [t for t in HW_ORDER.get(plat, []) if t in have]


def open_video(path: Path, hw: str | None):
    """(container, stream, decoder label) of `path`, on the hardware decoder `hw` when given."""
    import av

    kw = {}
    if hw:
        from av.codec.hwaccel import HWAccel

        kw["hwaccel"] = HWAccel(device_type=hw, allow_software_fallback=True)
    container = av.open(str(path), **kw)
    if not container.streams.video:
        container.close()
        fail(f"{path} has no video stream")
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    decoder = f"hw:{hw}" if hw and stream.codec_context.is_hwaccel else "sw"
    return container, stream, decoder


# After --max-frames, decoding goes on this long (seconds of video, else frames) to see whether
# the filter passes one more frame (capped) or the video ends (not capped).
LOOKAHEAD_S = 12.0
LOOKAHEAD_FRAMES = 360


def open_first(video: Path, candidates: list[str]):
    """Open `video` on the first hardware decoder of `candidates` that opens it, else in software."""
    for hw in candidates:
        try:
            return open_video(video, hw)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            log(f"hardware decoder {hw} could not open {video.name} ({e}); trying the next")
    try:
        return open_video(video, None)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        fail(f"cannot open {video}: {e} (is it a video file? Blue Iris .bvr clips must be exported to MP4 first)")


def extract(a) -> int:
    import av
    from PIL import Image  # noqa: F401  (frame.to_image needs Pillow)

    a.out.mkdir(parents=True, exist_ok=True)
    container, stream, decoder = open_first(a.video, hw_candidates(a.hwaccel))
    duration = None
    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
    elif container.duration:
        duration = container.duration / 1_000_000
    event("start", video=str(a.video), width=stream.codec_context.width, height=stream.codec_context.height,
          duration=duration, fps=float(stream.average_rate) if stream.average_rate else None,
          codec=stream.codec_context.name, decoder=decoder)
    # The filter graph is built on the first decoded frame: a hardware decoder's codec context
    # reports the device's pixel format, while the frames arrive downloaded (e.g. nv12), so the
    # buffer takes the frame's own size and format; a frame that differs (a software fallback,
    # a size change) is converted to it.
    graph = None
    gfmt = None
    kept = decoded = 0
    last_report = time.time()
    t0 = None
    last_t = None
    # The graph passed a frame beyond --max-frames (it is not written).
    over = False

    def build(frame):
        nonlocal graph, gfmt
        graph = av.filter.Graph()
        gfmt = (frame.width, frame.height, frame.format.name)
        src = graph.add_buffer(width=frame.width, height=frame.height, format=frame.format.name,
                               time_base=stream.time_base)
        sel = graph.add("select", a.select)
        sink = graph.add("buffersink")
        src.link_to(sel)
        sel.link_to(sink)
        graph.configure()

    def drain() -> None:
        """Write the frames the graph has, up to --max-frames; a frame beyond it sets `over`."""
        nonlocal kept, over
        while not over:
            try:
                f = graph.pull()
            except (av.error.BlockingIOError, av.error.EOFError, BlockingIOError, EOFError):
                return
            if kept >= a.max_frames:
                over = True
                return
            t = float(f.time) if f.time is not None else 0.0
            name = f"f_{kept:06d}.jpg"
            w, h = scaled_size(f.width, f.height, a.max_side)
            full = (w, h) == (f.width, f.height)
            img = f.to_image() if full else f.to_image(width=w, height=h, interpolation="AREA")
            save_jpeg(img, a.out / name)
            kept += 1
            # The select filter passes the frame just pushed: its decode index is decoded - 1.
            event("frame", file=name, t=round(t - (t0 or 0.0), 3), w=w, h=h, sw=f.width, sh=f.height,
                  pts=float(f.time) if f.time is not None else None, n=decoded - 1, full=full)

    # capped: the video had more to give than --max-frames: the graph passed a further frame, or
    # decoding stopped LOOKAHEAD_S (else LOOKAHEAD_FRAMES frames) after the quota was reached
    # with frames left in the stream (not examined).
    capped = False
    look_t = None
    look_n = 0
    # After a hardware failure the video is decoded again in software from the start: the frames
    # already pushed are skipped by time (`last_t`) when the frames carry one, else by count.
    skip_t = None
    skip_n = 0
    redecoded = 0
    frames = container.decode(stream)
    while True:
        try:
            frame = next(frames)
        except StopIteration:
            break
        except av.error.FFmpegError as e:
            if decoder == "sw":
                raise
            log(f"hardware decoding of {a.video.name} failed ({e}); decoding again in software")
            container.close()
            container, stream, decoder = open_video(a.video, None)
            event("decoder", decoder=decoder, reason=str(e)[:200])
            skip_t, skip_n, redecoded = last_t, decoded, 0
            frames = container.decode(stream)
            continue
        if skip_n:
            redecoded += 1
            if frame.time is not None and skip_t is not None:
                if float(frame.time) <= skip_t:
                    continue
            elif redecoded <= skip_n:
                continue
        if kept >= a.max_frames:
            # Look a little further to tell "full" from "exactly enough": a frame the graph
            # passes now is one too many; past the look-ahead the rest is left unread.
            if look_n == 0:
                look_t = None if frame.time is None else float(frame.time) + LOOKAHEAD_S
            look_n += 1
            if look_n > LOOKAHEAD_FRAMES or (look_t is not None and frame.time is not None
                                              and float(frame.time) > look_t):
                capped = True
                break
        decoded += 1
        if frame.time is not None:
            last_t = float(frame.time)
            if t0 is None:
                t0 = last_t
        if graph is None:
            build(frame)
        elif (frame.width, frame.height, frame.format.name) != gfmt:
            frame = frame.reformat(width=gfmt[0], height=gfmt[1], format=gfmt[2])
        graph.push(frame)
        drain()
        if over:
            capped = True
            break
        if time.time() - last_report > 2 and frame.time is not None:
            event("progress", t=round(float(frame.time) - (t0 or 0.0), 1), duration=duration, frames=kept)
            last_report = time.time()
    if not capped and graph is not None:
        try:
            graph.push(None)
        except Exception:  # noqa: BLE001 - flushing an empty graph
            pass
        drain()
        capped = over
    container.close()
    event("done", frames=kept, decoded=decoded, duration=duration, capped=capped, decoder=decoder)
    return 0


# `frames`: decode on (instead of seeking) when the next frame is at most this far ahead.
SEEK_GAP_S = 3.0
# Seek this much earlier again when a seek landed after the frame (an imprecise index).
SEEK_BACK_S = (0.0, 10.0, 60.0)


def plan_targets(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """The frames to extract again: (by time, sorted by pts; by decode index, sorted by n, for
    frames without a time). Entries without either are dropped (reported missing by the caller)."""
    by_pts = sorted((i for i in items if i.get("pts") is not None), key=lambda i: float(i["pts"]))
    by_n = sorted((i for i in items if i.get("pts") is None and i.get("n") is not None),
                  key=lambda i: int(i["n"]))
    return by_pts, by_n


def tolerance(fps) -> float:
    """How far a decoded frame's time may be from the recorded one and still be that frame: the
    same file decodes to the same timestamps, so a quarter frame, at most 10 ms."""
    if fps:
        return min(0.25 / float(fps), 0.01)
    return 0.01


def refetch(a) -> int:
    import av
    from PIL import Image  # noqa: F401  (frame.to_image needs Pillow)

    items = json.loads(a.list.read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    by_pts, by_n = plan_targets(items)
    planned = {id(i) for i in by_pts} | {id(i) for i in by_n}
    written = missing = 0

    def miss(item, reason: str) -> None:
        nonlocal missing
        missing += 1
        event("missing", name=item.get("name"), pts=item.get("pts"), n=item.get("n"), reason=reason)

    for item in items:
        if id(item) not in planned:
            miss(item, "no time and no frame index recorded")
    if not by_pts and not by_n:
        event("done", written=0, missing=missing)
        return 0
    hw = hw_candidates(a.hwaccel)
    container, stream, decoder = open_first(a.video, hw)
    tol = tolerance(stream.average_rate)
    event("start", video=str(a.video), frames=len(by_pts) + len(by_n), decoder=decoder)

    def save(item, frame) -> None:
        nonlocal written
        name = str(item["name"])
        if "/" in name or "\\" in name or ".." in name:
            miss(item, "bad file name")
            return
        save_jpeg(frame.to_image(), a.out / name)
        written += 1
        event("frame", name=name, pts=item.get("pts"), n=item.get("n"), w=frame.width, h=frame.height)

    def reopen_sw(e) -> None:
        nonlocal container, stream, decoder
        log(f"hardware decoding of {a.video.name} failed ({e}); decoding again in software")
        container.close()
        container, stream, decoder = open_video(a.video, None)
        event("decoder", decoder=decoder, reason=str(e)[:200])

    # By time: seek to the keyframe before each frame, or decode on to it when it is close.
    frames = None
    pending = None  # a decoded frame not yet matched (it may be the next target)
    pos = None  # time of the last decoded frame
    for item in by_pts:
        target = float(item["pts"])
        found = None
        attempts = list(SEEK_BACK_S)
        must_seek = frames is None or pos is None or target < pos - tol or target - pos > SEEK_GAP_S
        while found is None:
            if must_seek:
                if not attempts:
                    break
                back = attempts.pop(0)
                offset = max(0.0, target - back)
                try:
                    container.seek(int(offset / stream.time_base), backward=True, any_frame=False,
                                   stream=stream)
                except av.error.FFmpegError as e:
                    log(f"seek to {offset:.3f} s failed ({e}); decoding from the start")
                    container.seek(0, stream=stream)
                    attempts = []
                frames = container.decode(stream)
                pending = None
                first = True
            else:
                first = False
            try:
                while True:
                    frame = pending if pending is not None else next(frames)
                    pending = None
                    if frame.time is None:
                        continue
                    t = float(frame.time)
                    pos = t
                    if t < target - tol:
                        first = False
                        continue
                    if abs(t - target) <= tol:
                        found = frame
                    else:
                        # Past the frame: keep it for the next target.
                        pending = frame
                    break
            except StopIteration:
                frames = None
                pos = None
            except av.error.FFmpegError as e:
                if decoder == "sw":
                    raise
                reopen_sw(e)
                frames, pending, pos, must_seek = None, None, None, True
                attempts = list(SEEK_BACK_S)
                continue
            if found is not None:
                break
            if must_seek and first and attempts:
                # The seek landed after the frame: try again from further back.
                continue
            break
        if found is None:
            miss(item, f"no frame at {target:.3f} s")
        else:
            save(item, found)

    # By index: one pass from the start, counting decoded frames.
    if by_n:
        container.close()
        container, stream, decoder = open_first(a.video, hw)
        frames = container.decode(stream)
        idx = -1
        k = 0
        try:
            for frame in frames:
                idx += 1
                while k < len(by_n) and int(by_n[k]["n"]) < idx:
                    miss(by_n[k], "frame index not reached")
                    k += 1
                if k >= len(by_n):
                    break
                if int(by_n[k]["n"]) == idx:
                    save(by_n[k], frame)
                    k += 1
        except av.error.FFmpegError as e:
            log(f"decoding {a.video.name} by frame index failed ({e})")
        for item in by_n[k:]:
            miss(item, "the video ended before this frame")
    container.close()
    event("done", written=written, missing=missing, decoder=decoder)
    return 0


def self_test() -> int:
    """The pure helpers (no PyAV): sizes, disk-full errors, target planning."""
    assert scaled_size(1920, 1080, 640) == (640, 360)
    assert scaled_size(1080, 1920, 640) == (360, 640)
    assert scaled_size(3840, 2160, 640) == (640, 360)
    assert scaled_size(640, 480, 640) == (640, 480)
    assert scaled_size(320, 240, 640) == (320, 240)
    assert scaled_size(1920, 1080, 0) == (1920, 1080)
    assert scaled_size(5000, 3, 640) == (640, 1)
    assert is_disk_full(OSError(errno.ENOSPC, "No space left on device"))
    full = OSError(errno.ENOSPC, "x")
    try:
        try:
            raise full
        except OSError as e:
            raise RuntimeError("wrapped") from e
    except RuntimeError as e:
        assert is_disk_full(e)
    win = OSError(22, "There is not enough space on the disk")
    win.winerror = 112
    assert is_disk_full(win)
    assert not is_disk_full(OSError(errno.ENOENT, "missing"))
    assert not is_disk_full(ValueError("x"))
    items = [{"name": "a", "pts": 5.0, "n": 150}, {"name": "b", "pts": 1.5, "n": 45},
             {"name": "c", "pts": None, "n": 9}, {"name": "d", "pts": None, "n": 3}, {"name": "e"}]
    by_pts, by_n = plan_targets(items)
    assert [i["name"] for i in by_pts] == ["b", "a"]
    assert [i["name"] for i in by_n] == ["d", "c"]
    assert tolerance(30) == 0.25 / 30 and tolerance(None) == 0.01 and tolerance(1) == 0.01
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"import_footage_selftest_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        try:
            from PIL import Image
        except ImportError:
            Image = None
        if Image is not None:
            out = tmp / "x.jpg"
            save_jpeg(Image.new("RGB", (8, 8)), out)
            assert out.is_file() and not (tmp / ".x.jpg.part").exists()
    finally:
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
    print("[import_footage] self-test passed")
    return 0


def fetch_url(a) -> int:
    import yt_dlp

    a.out.mkdir(parents=True, exist_ok=True)
    state = {"last": 0.0}

    def hook(d):
        if d.get("status") == "downloading" and time.time() - state["last"] > 2:
            state["last"] = time.time()
            event("progress", bytes=d.get("downloaded_bytes"), total=d.get("total_bytes") or d.get("total_bytes_estimate"))

    h = a.max_height
    opts = {
        "format": f"bv*[height<={h}][ext=mp4]/bv*[height<={h}]/b[height<={h}]/wv*/w",
        "outtmpl": str(a.out / "video.%(ext)s"),
        "noplaylist": True,
        "max_filesize": a.max_bytes,
        "match_filter": yt_dlp.utils.match_filter_func("!is_live"),
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "progress_hooks": [hook],
        "restrictfilenames": True,
        "writethumbnail": False,
        "cachedir": False,
        "socket_timeout": 30,
        "retries": 3,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(a.url, download=True)
        except yt_dlp.utils.DownloadError as e:
            fail(f"yt-dlp could not download {a.url}: {e}")
        if info is None:
            fail(f"{a.url}: nothing to download (a live stream or a filtered format)")
        if info.get("_type") == "playlist":
            fail(f"{a.url} is a playlist; give the URL of one video")
        files = [p for p in a.out.iterdir() if p.is_file() and p.name.startswith("video.") and not p.name.endswith((".part", ".ytdl"))]
        if not files:
            fail(f"{a.url}: no file was written (larger than {a.max_bytes // 1_000_000} MB, or no video-only format up to {h} px)")
        keep = {k: info.get(k) for k in ("title", "uploader", "channel", "webpage_url", "extractor", "license", "duration", "upload_date", "id")}
        event("fetched", file=str(files[0]), size=files[0].stat().st_size, **keep)
    return 0


def embed(a) -> int:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    paths = [l.strip() for l in a.list.read_text(encoding="utf-8").splitlines() if l.strip()]
    dev = a.device
    if dev == "auto":
        dev = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(a.model))
    model = AutoModel.from_pretrained(str(a.model)).to(dev).eval()
    event("start", images=len(paths), device=dev, load_s=round(time.time() - t0, 1))
    dim = None
    with open(a.out, "wb") as out:
        for i in range(0, len(paths), a.batch):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + a.batch]]
            inputs = processor(images=batch, return_tensors="pt").to(dev)
            with torch.no_grad():
                f = model.get_image_features(**inputs)
            f = getattr(f, "pooler_output", f)
            f = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy().astype("<f4")
            dim = f.shape[1]
            out.write(np.ascontiguousarray(f).tobytes())
            event("progress", done=min(len(paths), i + a.batch), images=len(paths))
    event("done", images=len(paths), dim=dim, seconds=round(time.time() - t0, 1), device=dev)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--self-test", action="store_true", help="test the pure helpers (no PyAV)")
    sub = p.add_subparsers(dest="cmd")
    e = sub.add_parser("extract")
    e.add_argument("--video", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--select", required=True, help="FFmpeg select expression")
    e.add_argument("--max-frames", type=int, default=2000)
    e.add_argument("--hwaccel", default="auto",
                   help="auto (the platform's hardware decoder when available), off, or a device type")
    e.add_argument("--max-side", type=int, default=0,
                   help="store the frames at most this many px on the long side (0 = full resolution)")
    r = sub.add_parser("frames")
    r.add_argument("--video", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--list", type=Path, required=True, help='JSON: [{"name", "pts", "n"}]')
    r.add_argument("--hwaccel", default="auto")
    f = sub.add_parser("fetch-url")
    f.add_argument("--url", required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--max-bytes", type=int, default=2_000_000_000)
    f.add_argument("--max-height", type=int, default=1080)
    m = sub.add_parser("embed")
    m.add_argument("--list", type=Path, required=True)
    m.add_argument("--model", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--device", default="auto")
    m.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    if not a.self_check and not a.self_test and not a.cmd:
        p.error("give a command (extract, frames, fetch-url, embed), --self-check or --self-test")
    return a


def main() -> int:
    a = parse_args()
    if a.self_test:
        return self_test()
    if a.self_check:
        return self_check()
    watch_parent()
    if a.cmd == "extract":
        return extract(a)
    if a.cmd == "frames":
        return refetch(a)
    if a.cmd == "fetch-url":
        return fetch_url(a)
    return embed(a)


if __name__ == "__main__":
    sys.exit(main())
