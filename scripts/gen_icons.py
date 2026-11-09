#!/usr/bin/env python
"""Home Screen and manifest icons of Blue Onyx Prism, drawn from the favicon's design.

assets/favicon.ico is a 32x32 picture: a dark navy disc inside a mid-blue ring, with a round
light-blue highlight upper left. This script draws the same shapes as circles at any size (no
upscaled pixels) and writes them to assets/icons/:

  apple-touch-icon.png   180x180, opaque: iOS rounds the corners itself and shows black where a
                         pixel is transparent, so the square is filled with the header colour and
                         the disc sits inside it.
  icon-192.png           192x192, transparent outside the disc (manifest, purpose "any").
  icon-512.png           512x512, the same.
  icon-maskable-512.png  512x512, opaque, the disc inside the central 80% safe zone (purpose
                         "maskable": launchers crop it to their own shape).

Edges are anti-aliased by supersampling: a pixel near an edge is the mean of 8x8 samples, any other
pixel is one sample (the same answer, since no edge crosses it). Python standard library only
(zlib, struct), no Pillow. The PNG bytes are deterministic for a given zlib (level 9, filter type
0 on every row, no time or text chunks).

Usage:
  python3 -I scripts/gen_icons.py           write the icons
  python3 -I scripts/gen_icons.py --check   exit 1 when a committed icon's pixels differ from what
                                            this script draws (decoded and compared, so another zlib
                                            build's compression does not count), naming it
"""

import argparse
import os
import struct
import sys
import zlib

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR = os.path.join(ROOT, "assets", "icons")

# The favicon's colours (read from its pixels) and the page header's (`--head` in style.css).
RING = (0x1F, 0x5F, 0xBF)  # --accent
DISC = (0x12, 0x18, 0x26)
HIGHLIGHT = (0x78, 0xAA, 0xF0)
BACKDROP = (0x0F, 0x27, 0x47)  # --head

# The favicon's geometry on its 32 px grid, as fractions of the disc's outer radius: outer edge
# 15 px from the centre, inner (navy) edge 12 px, highlight radius 4 px centred 4 px left of and
# 4 px above the centre.
INNER = 12.0 / 15.0
HIGHLIGHT_R = 4.0 / 15.0
HIGHLIGHT_OFFSET = -4.0 / 15.0

# Samples per axis on a pixel an edge crosses.
SUPERSAMPLE = 8


class Icon:
    def __init__(self, name, size, radius, backdrop):
        self.name = name
        self.size = size
        # Outer radius of the disc as a fraction of the side.
        self.radius = radius
        # None: transparent outside the disc (RGBA); else the square's colour (RGB).
        self.backdrop = backdrop


ICONS = [
    # The disc at 80% of the side: iOS's rounded corners never reach it.
    Icon("apple-touch-icon.png", 180, 0.40, BACKDROP),
    # As the favicon: 1/32 of the side free around the disc.
    Icon("icon-192.png", 192, 15.0 / 32.0, None),
    Icon("icon-512.png", 512, 15.0 / 32.0, None),
    # The safe zone is the centred circle of radius 40% of the side; the disc keeps inside it.
    Icon("icon-maskable-512.png", 512, 0.36, BACKDROP),
]


def shapes(icon):
    """The circles, bottom first: (centre x, centre y, radius, colour) in pixels."""
    c = icon.size / 2.0
    r = icon.radius * icon.size
    h = c + HIGHLIGHT_OFFSET * r
    return [
        (c, c, r, RING),
        (c, c, INNER * r, DISC),
        (h, h, HIGHLIGHT_R * r, HIGHLIGHT),
    ]


def colour_at(x, y, circles, backdrop):
    """(r, g, b, a) of the point (x, y): the topmost circle holding it, else the backdrop."""
    for cx, cy, r, colour in reversed(circles):
        dx, dy = x - cx, y - cy
        if dx * dx + dy * dy <= r * r:
            return colour + (255,)
    if backdrop is None:
        return (0, 0, 0, 0)
    return backdrop + (255,)


def near_edge(x, y, circles):
    """Whether an edge may cross the pixel whose centre is (x, y): within one pixel diagonal."""
    for cx, cy, r, _ in circles:
        dx, dy = x - cx, y - cy
        d2 = dx * dx + dy * dy
        lo, hi = max(r - 0.75, 0.0), r + 0.75
        if lo * lo <= d2 <= hi * hi:
            return True
    return False


def render(icon):
    """Rows of the icon's pixels: bytes of RGBA (no backdrop) or RGB (backdrop)."""
    circles = shapes(icon)
    n = SUPERSAMPLE
    offsets = [(i + 0.5) / n for i in range(n)]
    channels = 3 if icon.backdrop is not None else 4
    rows = []
    for py in range(icon.size):
        row = bytearray()
        for px in range(icon.size):
            if not near_edge(px + 0.5, py + 0.5, circles):
                pixel = colour_at(px + 0.5, py + 0.5, circles, icon.backdrop)
            else:
                # Premultiplied sums, so transparent samples add no colour.
                sr = sg = sb = sa = 0
                for oy in offsets:
                    for ox in offsets:
                        r, g, b, a = colour_at(px + ox, py + oy, circles, icon.backdrop)
                        sr += r * a
                        sg += g * a
                        sb += b * a
                        sa += a
                if sa == 0:
                    pixel = (0, 0, 0, 0)
                else:
                    # Rounded integer division: the same bytes on every platform.
                    pixel = (
                        (2 * sr + sa) // (2 * sa),
                        (2 * sg + sa) // (2 * sa),
                        (2 * sb + sa) // (2 * sa),
                        (2 * sa + n * n) // (2 * n * n),
                    )
            row.extend(pixel[:channels])
        rows.append(bytes(row))
    return rows


def chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def ihdr(icon):
    colour_type = 2 if icon.backdrop is not None else 6
    return struct.pack(">IIBBBBB", icon.size, icon.size, 8, colour_type, 0, 0, 0)


def encode(icon, rows):
    raw = b"".join(b"\x00" + row for row in rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr(icon))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def decode(data):
    """(IHDR bytes, raw scanlines) of a PNG this script wrote (filter type 0 rows); None when the
    file is not one."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    pos, header, idat = 8, None, b""
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            header = body
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
    if header is None:
        return None
    try:
        return header, zlib.decompress(idat)
    except zlib.error:
        return None


def check():
    stale = []
    for icon in ICONS:
        path = os.path.join(OUT_DIR, icon.name)
        try:
            with open(path, "rb") as f:
                committed = decode(f.read())
        except OSError:
            committed = None
        want = (ihdr(icon), b"".join(b"\x00" + row for row in render(icon)))
        if committed != want:
            stale.append(icon.name)
    if stale:
        for name in stale:
            print(f"assets/icons/{name} differs from what scripts/gen_icons.py draws", file=sys.stderr)
        print("run: python3 -I scripts/gen_icons.py, then commit assets/icons/", file=sys.stderr)
        return 1
    print(f"icons ok ({len(ICONS)} files)")
    return 0


def write():
    os.makedirs(OUT_DIR, exist_ok=True)
    for icon in ICONS:
        data = encode(icon, render(icon))
        path = os.path.join(OUT_DIR, icon.name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"assets/icons/{icon.name}: {icon.size}x{icon.size}, {len(data)} bytes")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the committed icons with a fresh drawing; exit 1 on a difference",
    )
    args = parser.parse_args()
    return check() if args.check else write()


if __name__ == "__main__":
    sys.exit(main())
