"""Little-endian reader with WZ compressed integers and encrypted strings (port of `WzBinaryReader.cs`)."""

from __future__ import annotations

import struct

from wz.crypto import WzKey

_I8 = struct.Struct("<b")
_I16 = struct.Struct("<h")
_U16 = struct.Struct("<H")
_I32 = struct.Struct("<i")
_U32 = struct.Struct("<I")
_I64 = struct.Struct("<q")
_F32 = struct.Struct("<f")
_F64 = struct.Struct("<d")


class WzReader:
    """Cursor over a bytes-like buffer.

    Parameters
    ----------
    data
        Backing buffer (`bytes`, `memoryview`, or `mmap`).
    pos
        Initial absolute position.
    base
        Position that image-relative string offsets are measured from.
    """

    __slots__ = ("base", "data", "pos", "string_cache")

    def __init__(self, data: bytes | memoryview, pos: int = 0, base: int = 0) -> None:
        self.data = data
        self.pos = pos
        self.base = base
        self.string_cache: dict[int, str] = {}

    def _unpack(self, fmt: struct.Struct) -> int | float:
        value = fmt.unpack_from(self.data, self.pos)[0]
        self.pos += fmt.size
        return value

    def u8(self) -> int:
        """Read an unsigned byte."""
        value = self.data[self.pos]
        self.pos += 1
        return value

    def i8(self) -> int:
        """Read a signed byte."""
        return int(self._unpack(_I8))

    def i16(self) -> int:
        """Read a signed 16-bit int."""
        return int(self._unpack(_I16))

    def u16(self) -> int:
        """Read an unsigned 16-bit int."""
        return int(self._unpack(_U16))

    def i32(self) -> int:
        """Read a signed 32-bit int."""
        return int(self._unpack(_I32))

    def u32(self) -> int:
        """Read an unsigned 32-bit int."""
        return int(self._unpack(_U32))

    def i64(self) -> int:
        """Read a signed 64-bit int."""
        return int(self._unpack(_I64))

    def f64(self) -> float:
        """Read a double."""
        return float(self._unpack(_F64))

    def cint(self) -> int:
        """Read a WZ compressed int32 (1 byte, or `0x80` followed by an int32)."""
        s = self.i8()
        return self.i32() if s == -128 else s

    def clong(self) -> int:
        """Read a WZ compressed int64."""
        s = self.i8()
        return self.i64() if s == -128 else s

    def cfloat(self) -> float:
        """Read a WZ compressed float (compressed int32 bit pattern)."""
        bits = self.cint()
        return float(_F32.unpack(_I32.pack(bits))[0])

    def read(self, n: int) -> bytes:
        """Read `n` raw bytes."""
        out = bytes(self.data[self.pos : self.pos + n])
        if len(out) != n:
            raise EOFError(f"Wanted {n} bytes at {self.pos}, got {len(out)}")
        self.pos += n
        return out

    def skip(self, n: int) -> None:
        """Advance by `n` bytes."""
        self.pos += n

    def string(self, key: WzKey) -> str:
        """Read an inline length-prefixed encrypted string."""
        start = self.pos
        cached = self.string_cache.get(start)
        size = self.i8()
        if size < 0:
            n = self.i32() if size == -128 else -size
            if cached is not None:
                self.pos += n
                return cached
            value = key.decode_ascii(self.read(n))
        elif size > 0:
            n = self.i32() if size == 127 else size
            if cached is not None:
                self.pos += n * 2
                return cached
            value = key.decode_utf16(self.read(n * 2))
        else:
            return ""
        self.string_cache[start] = value
        return value

    def string_at(self, offset: int, key: WzKey) -> str:
        """Read a string at absolute `offset` without moving the cursor."""
        cached = self.string_cache.get(offset)
        if cached is not None:
            return cached
        saved = self.pos
        self.pos = offset
        try:
            return self.string(key)
        finally:
            self.pos = saved

    def image_string(self, key: WzKey) -> str | None:
        """Read a property-name/string value: inline (`0x00`) or back-reference (`0x01`)."""
        flag = self.u8()
        if flag == 0x00:
            return self.string(key)
        if flag == 0x01:
            return self.string_at(self.base + self.i32(), key)
        if flag == 0x04:
            self.skip(8)
            return None
        raise ValueError(f"Unexpected string flag 0x{flag:02x} at {self.pos - 1}")

    def object_type_name(self, key: WzKey) -> str:
        """Read the type tag of an embedded object: inline (`0x73`) or back-reference (`0x1B`)."""
        flag = self.u8()
        if flag == 0x73:
            return self.string(key)
        if flag == 0x1B:
            return self.string_at(self.base + self.i32(), key)
        raise ValueError(f"Unexpected object tag flag 0x{flag:02x} at {self.pos - 1}")
