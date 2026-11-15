"""Canvas pixel decoding to PIL images (port of `Wz_Png.cs` and parts of `ImageCodec.cs`)."""

from __future__ import annotations

import enum
import struct
import zlib
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from wz.crypto import WzKey
    from wz.values import WzCanvas


class TextureFormat(enum.IntEnum):
    """Texture formats used by WZ canvases."""

    ARGB4444 = 1
    ARGB8888 = 2
    ARGB1555 = 257
    RGB565 = 513
    R16 = 769
    DXT3 = 1026
    DXT5 = 2050
    A8 = 2304
    RGBA1010102 = 2562
    DXT1 = 4097
    BC7 = 4098
    RGBA32FLOAT = 4100


_BCN = {TextureFormat.DXT1: 1, TextureFormat.DXT3: 2, TextureFormat.DXT5: 3, TextureFormat.BC7: 7}


def raw_size(fmt: int, width: int, height: int) -> int:
    """Uncompressed size in bytes of one page of `fmt` pixels at `width` x `height`."""
    if fmt in (TextureFormat.ARGB4444, TextureFormat.ARGB1555, TextureFormat.RGB565, TextureFormat.R16):
        return width * height * 2
    if fmt in (TextureFormat.ARGB8888, TextureFormat.RGBA1010102):
        return width * height * 4
    if fmt in (TextureFormat.DXT3, TextureFormat.DXT5):
        return ((width + 3) // 4) * ((height + 3) // 4) * 16
    if fmt == TextureFormat.DXT1:
        return ((width + 3) // 4) * ((height + 3) // 4) * 8
    if fmt == TextureFormat.BC7:
        return width * (height & ~3)
    if fmt == TextureFormat.A8:
        return width * height
    if fmt == TextureFormat.RGBA32FLOAT:
        return width * height * 16
    raise ValueError(f"Unknown texture format {fmt}")


def _is_zlib_header(b0: int, b1: int) -> bool:
    return (b0 & 0x0F) == 8 and (b0 >> 4) <= 7 and ((b0 << 8) | b1) % 31 == 0


def _dechunk(payload: bytes, key: WzKey) -> bytes:
    out = bytearray()
    pos = 0
    n = len(payload)
    while pos + 4 <= n:
        (length,) = struct.unpack_from("<i", payload, pos)
        pos += 4
        if length <= 0 or pos + length > n:
            break
        out += key.xor(payload[pos : pos + length])
        pos += length
    return bytes(out)


def decompress_canvas(canvas: WzCanvas, max_bytes: int | None = None) -> bytes:
    """Return the decompressed pixel bytes of `canvas` (all pages).

    Parameters
    ----------
    canvas
        Canvas to read.
    max_bytes
        Stop after this many output bytes.
    """
    data, _ = canvas.image.buffer()
    start = canvas.data_offset + 1  # the first byte is a flag
    payload = bytes(data[start : canvas.data_offset + canvas.data_length])
    if len(payload) >= 2 and _is_zlib_header(payload[0], payload[1]):
        stream = payload[2:]
    else:
        key = canvas.image.key
        if key is None:
            raise ValueError("Encrypted canvas data requires a detected image key")
        stream = _dechunk(payload, key)[2:]
    d = zlib.decompressobj(-15)
    return d.decompress(stream, max_bytes or 0) + (d.flush() if not max_bytes else b"")


def _expand(v: np.ndarray, bits: int) -> np.ndarray:
    return ((v << (8 - bits)) | (v >> (2 * bits - 8))).astype(np.uint8) if bits < 8 else v.astype(np.uint8)


def _decode_pixels(fmt: int, raw: bytes, w: int, h: int) -> Image.Image:
    if fmt == TextureFormat.ARGB8888:
        return Image.frombytes("RGBA", (w, h), raw, "raw", "BGRA")
    if fmt == TextureFormat.ARGB4444:
        p = np.frombuffer(raw, dtype="<u2", count=w * h).reshape(h, w)
        rgba = np.stack([(p >> 8) & 0xF, (p >> 4) & 0xF, p & 0xF, p >> 12], axis=-1).astype(np.uint8) * 17
        return Image.fromarray(rgba, "RGBA")
    if fmt == TextureFormat.ARGB1555:
        p = np.frombuffer(raw, dtype="<u2", count=w * h).reshape(h, w)
        a = np.where(p >> 15, 255, 0).astype(np.uint8)
        rgba = np.stack([_expand((p >> 10) & 31, 5), _expand((p >> 5) & 31, 5), _expand(p & 31, 5), a], axis=-1)
        return Image.fromarray(rgba, "RGBA")
    if fmt == TextureFormat.RGB565:
        p = np.frombuffer(raw, dtype="<u2", count=w * h).reshape(h, w)
        a = np.full((h, w), 255, dtype=np.uint8)
        rgba = np.stack([_expand(p >> 11, 5), _expand((p >> 5) & 63, 6), _expand(p & 31, 5), a], axis=-1)
        return Image.fromarray(rgba, "RGBA")
    if fmt == TextureFormat.R16:
        p = np.frombuffer(raw, dtype="<u2", count=w * h).reshape(h, w)
        z = np.zeros((h, w), dtype=np.uint8)
        rgba = np.stack([(p >> 8).astype(np.uint8), z, z, np.full((h, w), 255, dtype=np.uint8)], axis=-1)
        return Image.fromarray(rgba, "RGBA")
    if fmt == TextureFormat.A8:
        a = np.frombuffer(raw, dtype=np.uint8, count=w * h).reshape(h, w)
        white = np.full((h, w), 255, dtype=np.uint8)
        return Image.fromarray(np.stack([white, white, white, a], axis=-1), "RGBA")
    if fmt == TextureFormat.RGBA1010102:
        p = np.frombuffer(raw, dtype="<u4", count=w * h).reshape(h, w)
        rgba = np.stack(
            [(p & 0x3FF) >> 2, ((p >> 10) & 0x3FF) >> 2, ((p >> 20) & 0x3FF) >> 2, (p >> 30) * 85], axis=-1
        ).astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")
    if fmt == TextureFormat.RGBA32FLOAT:
        p = np.frombuffer(raw, dtype="<f4", count=w * h * 4).reshape(h, w, 4)
        return Image.fromarray((np.clip(p, 0.0, 1.0) * 255).astype(np.uint8), "RGBA")
    if fmt in _BCN:
        if fmt == TextureFormat.BC7:
            bw = (w * 4 // 16) * 4
            bh = h & ~3
        else:
            bw = (w + 3) & ~3
            bh = (h + 3) & ~3
        img = Image.frombytes("RGBA", (max(bw, 4), max(bh, 4)), raw, "bcn", _BCN[TextureFormat(fmt)])
        out = Image.new("RGBA", (w, h))
        out.paste(img.crop((0, 0, min(w, bw), min(h, bh))), (0, 0))
        return out
    raise ValueError(f"Unsupported texture format {fmt}")


def decode_canvas(canvas: WzCanvas, page: int = 0) -> Image.Image:
    """Decode one page of `canvas` to an RGBA PIL image.

    Parameters
    ----------
    canvas
        Canvas to decode.
    page
        Page index (ignored for single-page canvases).

    Returns
    -------
    The bitmap at the canvas' nominal `width` x `height`.
    """
    scale = 1 << canvas.scale if canvas.scale > 0 else 1
    w = max(canvas.width // scale, 1)
    h = max(canvas.height // scale, 1)
    per_page = raw_size(canvas.format, w, h)
    page = page if canvas.pages > 0 else 0
    raw = decompress_canvas(canvas, per_page * (page + 1))
    raw = raw[per_page * page : per_page * (page + 1)]
    if len(raw) < per_page:
        raise ValueError(f"Canvas data too short: {len(raw)} < {per_page} bytes")
    img = _decode_pixels(canvas.format, raw, w, h)
    if scale > 1:
        img = img.resize((canvas.width, canvas.height), Image.Resampling.NEAREST)
    return img
