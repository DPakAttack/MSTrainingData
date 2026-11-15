"""Lazy `.img` property parser (port of `Wz_Image.cs`)."""

from __future__ import annotations

from collections.abc import Callable

from wz.binary_reader import WzReader
from wz.crypto import WzKey, all_keys
from wz.node import WzNode
from wz.values import WzCanvas, WzConvex, WzRawData, WzSound, WzUol, WzVector, WzVideo

_KNOWN_TAGS = frozenset(
    {"Property", "Shape2D#Vector2D", "Canvas", "Shape2D#Convex2D", "Sound_DX8", "UOL", "RawData", "Canvas#Video"}
)


class KeyHint:
    """Shared, mutable "last key that worked" so detection usually succeeds on the first try."""

    def __init__(self) -> None:
        self.key: WzKey | None = None

    def candidates(self) -> list[WzKey]:
        """Keys to try, preferred first."""
        keys = all_keys()
        if self.key is not None:
            keys.remove(self.key)
            keys.insert(0, self.key)
        return keys


class WzImage:
    """One `.img` blob inside a `.wz` or `.ms` package.

    Parameters
    ----------
    name
        Image file name, e.g. `0100100.img`.
    loader
        Returns `(buffer, start)`: a buffer containing the image and the image's start offset in it.
    size
        Image size in bytes.
    key_hint
        Shared key preference used during encryption detection.
    """

    def __init__(
        self,
        name: str,
        loader: Callable[[], tuple[bytes | memoryview, int]],
        size: int,
        key_hint: KeyHint,
    ) -> None:
        self.name = name
        self.size = size
        self._loader = loader
        self._key_hint = key_hint
        self._buffer: bytes | memoryview | None = None
        self._start = 0
        self.key: WzKey | None = None

    def buffer(self) -> tuple[bytes | memoryview, int]:
        """Return `(buffer, start)` for the image bytes, loading them if needed."""
        if self._buffer is None:
            self._buffer, self._start = self._loader()
        return self._buffer, self._start

    def release(self) -> None:
        """Forget cached image bytes (they are re-read on next access)."""
        self._buffer = None

    def _detect_key(self) -> WzKey:
        if self.key is not None:
            return self.key
        data, start = self.buffer()
        for key in self._key_hint.candidates():
            try:
                tag = WzReader(data, start, start).object_type_name(key)
            except (ValueError, IndexError, EOFError, UnicodeDecodeError):
                continue
            if tag in _KNOWN_TAGS:
                self.key = key
                self._key_hint.key = key
                return key
        raise ValueError(f"Could not detect string encryption for image {self.name}")

    def extract(self, node: WzNode) -> None:
        """Parse all properties of this image into `node`."""
        key = self._detect_key()
        data, start = self.buffer()
        reader = WzReader(data, start, start)
        _Parser(reader, key, self).read_object(node)


class _Parser:
    def __init__(self, reader: WzReader, key: WzKey, image: WzImage) -> None:
        self.r = reader
        self.key = key
        self.image = image

    def read_property_list(self, parent: WzNode) -> None:
        self.r.skip(2)
        for _ in range(self.r.cint()):
            self.read_value(parent)

    def read_object(self, parent: WzNode) -> None:
        r = self.r
        tag = r.object_type_name(self.key)
        if tag == "Property":
            self.read_property_list(parent)
        elif tag == "Shape2D#Vector2D":
            parent.value = WzVector(r.cint(), r.cint())
        elif tag == "Canvas":
            r.skip(1)
            if r.u8() == 0x01:
                self.read_property_list(parent)
            width = r.cint()
            height = r.cint()
            fmt = r.cint()
            scale = r.u8()
            pages = r.cint()
            r.cint()  # unknown1, introduced in KMST1198
            r.skip(2)
            data_len = r.i32()
            parent.value = WzCanvas(width, height, fmt, scale, pages, r.pos, data_len, self.image)
            r.skip(data_len)
        elif tag == "Shape2D#Convex2D":
            points = []
            for _ in range(r.cint()):
                tmp = WzNode("")
                self.read_object(tmp)
                if not isinstance(tmp.value, WzVector):
                    raise ValueError("Convex2D contains a non-vector item")
                points.append(tmp.value)
            parent.value = WzConvex(tuple(points))
        elif tag == "Sound_DX8":
            self._read_sound(parent)
        elif tag == "UOL":
            r.skip(1)
            parent.value = WzUol(r.image_string(self.key) or "")
        elif tag == "RawData":
            if r.u8() == 1 and r.u8() == 0x01:
                self.read_property_list(parent)
            length = r.cint()
            parent.value = WzRawData(r.pos, length)
            r.skip(length)
        elif tag == "Canvas#Video":
            r.skip(1)
            if r.u8() == 0x01:
                self.read_property_list(parent)
            r.skip(1)
            length = r.cint()
            parent.value = WzVideo(r.pos, length)
            r.skip(length)
        else:
            raise ValueError(f"Unknown object tag {tag!r} in {self.image.name}")

    def _read_sound(self, parent: WzNode) -> None:
        r = self.r
        if r.u8() == 1 and r.u8() == 0x01:
            self.read_property_list(parent)
        data_len = r.cint()
        duration = r.cint()
        sound_decl = r.u8()
        r.skip(16 + 16 + 1 + 1 + 16)  # major type, subtype, flags, format type
        if sound_decl == 2:
            r.skip(r.cint())
        parent.value = WzSound(r.pos, data_len, duration)
        r.skip(data_len)

    def read_value(self, parent: WzNode) -> None:
        r = self.r
        name = r.image_string(self.key)
        node = parent.add(name if name is not None else "")
        flag = r.u8()
        if flag == 0x00:
            node.value = None
        elif flag == 0x02:
            node.value = r.i16()
        elif flag == 0x03:
            node.value = r.cint()
        elif flag == 0x0B:
            node.value = r.i16() != 0
        elif flag == 0x04:
            node.value = r.cfloat()
        elif flag == 0x05:
            node.value = r.f64()
        elif flag == 0x08:
            node.value = r.image_string(self.key)
        elif flag == 0x09:
            length = r.i32()
            end = r.pos + length
            self.read_object(node)
            r.pos = end
        elif flag == 0x13:
            node.value = r.cint() & 0xFFFFFFFF
        elif flag == 0x14:
            node.value = r.clong()
        else:
            raise ValueError(f"Unknown value type 0x{flag:02x} at {r.pos - 1} in {self.image.name}")
