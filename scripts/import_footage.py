#!/usr/bin/env python
"""Footage import helpers of Blue Onyx Prism (`blue-onyx-prism train import`).

The service plans an import and writes the frames into the dataset folder itself; this script only
does what needs the training environment's libraries:

  extract    Decode a video with PyAV (its bundled FFmpeg, LGPL) and keep the frames FFmpeg's
             `select` filter passes (--select: the expression the service built from the scene-change
             threshold, the maximum frame rate and the keep-one-every interval), at most
             --max-frames. Writes f_<n>.jpg (quality 92) into --out and one `frame` event per frame
             (file, time in seconds, size). --hwaccel auto decodes on the platform's hardware
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
  fetch-url  Download one video with yt-dlp (no playlists, no live streams, video-only formats up to
             --max-height so no ffmpeg binary is needed to merge streams, at most --max-bytes).
             Prints a `fetched` event with the file and what the site reports (title, uploader,
             licence, page URL) for the provenance record. The service checked the URL against its
             denylist and the user confirmed the licence before this runs.
  embed      SigLIP 2 image embeddings (L2-normalized float32, row-major) of the images listed in
             --list, written to --out; the service picks the frames (k-center greedy) from them.

Progress: one stdout line `@import {json}` per event; everything else is log output.

    python scripts/import_footage.py --self-check
    python scripts/import_footage.py extract --video clip.mp4 --out frames/ --hwaccel auto \\
        --select "if(isnan(prev_selected_t),1,gt(scene,0.3)*gte(t-prev_selected_t,0.5)+gte(t-prev_selected_t,10))"
    python scripts/import_footage.py fetch-url --url https://example.org/v.mp4 --out dl/
    python scripts/import_footage.py embed --list frames.txt --model <siglip2 dir> --out emb.f32
"""

from __future__ import annotations

import argparse
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
            f.to_image().save(a.out / name, quality=92)
            kept += 1
            event("frame", file=name, t=round(t - (t0 or 0.0), 3), w=f.width, h=f.height)

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
    sub = p.add_subparsers(dest="cmd")
    e = sub.add_parser("extract")
    e.add_argument("--video", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--select", required=True, help="FFmpeg select expression")
    e.add_argument("--max-frames", type=int, default=2000)
    e.add_argument("--hwaccel", default="auto",
                   help="auto (the platform's hardware decoder when available), off, or a device type")
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
    if not a.self_check and not a.cmd:
        p.error("give a command (extract, fetch-url, embed) or --self-check")
    return a


def main() -> int:
    a = parse_args()
    if a.self_check:
        return self_check()
    watch_parent()
    if a.cmd == "extract":
        return extract(a)
    if a.cmd == "fetch-url":
        return fetch_url(a)
    return embed(a)


if __name__ == "__main__":
    sys.exit(main())
