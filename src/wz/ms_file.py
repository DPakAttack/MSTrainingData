"""`.ms` pack readers (port of `Ms_File.cs`/`Ms_Image.cs` and `Ms_FileV2.cs`/`Ms_ImageV2.cs`).

An `.ms` file is an encrypted archive of `.img` blobs whose entry names are full paths such as
`Mob/0100100.img`. Two layouts exist:

- v1 (header version 2): Snow2-encrypted header, entry table, and image data.
- v2 (header version 4): ChaCha20-encrypted header and entry table; only the first 1024 bytes of
  each image are encrypted.

Use `open_ms_file` to pick the right one.
"""

from __future__ import annotations

import logging
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from wz.snow2 import Snow2
from wz.wz_image import KeyHint, WzImage

logger = logging.getLogger(__name__)

_CHACHA_KEY_OBSCURE = bytes(
    [
        0x7B, 0x2F, 0x35, 0x48, 0x43, 0x95, 0x02, 0xB9, 0xAE, 0x91, 0xA6, 0xE1, 0xD8, 0xD6, 0x24, 0xB4,
        0x33, 0x10, 0x1D, 0x3D, 0xC1, 0xBB, 0xC6, 0xF4, 0xA5, 0xFE, 0xB3, 0x69, 0x6B, 0x56, 0xE4, 0x75,
    ]
)  # fmt: skip


@dataclass
class MsEntry:
    """One encrypted image in an `.ms` pack."""

    name: str
    """Full entry path, e.g. `Mob/0100100.img`."""

    checksum: int
    """Stored checksum."""

    start: int
    """Absolute file offset of the encrypted data."""

    size: int
    """Plain data size in bytes."""

    size_aligned: int
    """Stored (1024-aligned) data size in bytes."""

    key: bytes
    """Per-entry 16-byte key material."""


def _to_i32(v: int) -> int:
    return ((v + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def _fnv1a(text: str) -> int:
    h = 0x811C9DC5
    for c in text:
        h = ((h ^ ord(c)) * 0x1000193) & 0xFFFFFFFF
    return h


def _entry_key(entry: MsEntry, key_hash: int, length: int) -> bytes:
    digits = [ord(c) - ord("0") for c in str(key_hash)]
    nd = len(digits)
    name = entry.name
    ek = entry.key
    return bytes(
        (
            i
            + ord(name[i % len(name)])
            * (digits[i % nd] % 2 + ek[(digits[(i + 2) % nd] + i) % len(ek)] + (digits[(i + 1) % nd] + i) % 5)
        )
        & 0xFF
        for i in range(length)
    )


def _chacha20(key: bytes, nonce: bytes, counter: int, data: bytes) -> bytes:
    full_nonce = struct.pack("<I", counter & 0xFFFFFFFF) + nonce
    return Cipher(algorithms.ChaCha20(key, full_nonce), mode=None).encryptor().update(data)


class MsFileBase:
    """Shared plumbing for both `.ms` layouts."""

    def __init__(self, path: Path, key_hint: KeyHint) -> None:
        self.path = path
        self._key_hint = key_hint
        self._fh = path.open("rb")
        try:
            self.data = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._fh.close()
            raise
        self._file_name = path.name.lower()
        self.salt = ""
        self.entries: list[MsEntry] = []
        try:
            self._read()
        except Exception:
            self.close()
            raise

    def _read(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Close the memory map and file handle.

        If views into the map are still alive, the map is left for garbage collection.
        """
        try:
            self.data.close()
        except BufferError:
            pass
        self._fh.close()

    def read_entry(self, entry: MsEntry) -> bytes:
        """Decrypt and return the plain bytes of `entry`."""
        raise NotImplementedError

    def make_image(self, entry: MsEntry) -> WzImage:
        """Create a lazily-decrypted `WzImage` for `entry`."""

        def loader() -> tuple[bytes, int]:
            return self.read_entry(entry), 0

        return WzImage(entry.name.rsplit("/", 1)[-1], loader, entry.size, self._key_hint)

    @staticmethod
    def _align_1k(pos: int) -> int:
        return pos if not pos & 0x3FF else pos - (pos & 0x3FF) + 0x400


class MsFileV1(MsFileBase):
    """`.ms` layout with header version 2 (Snow2 everywhere)."""

    def _read(self) -> None:
        name = self._file_name
        data = self.data
        rand_count = sum(ord(c) for c in name) % 312 + 30
        rand = data[:rand_count]
        pos = rand_count
        (hashed_salt_len,) = struct.unpack_from("<i", data, pos)
        pos += 4
        salt_len = (hashed_salt_len & 0xFF) ^ rand[0]
        salt_bytes = bytes(data[pos : pos + salt_len * 2])
        pos += salt_len * 2
        self.salt = "".join(chr(rand[i] ^ salt_bytes[i * 2]) for i in range(salt_len))
        nws = name + self.salt

        key = bytes((ord(nws[i % len(nws)]) + i) & 0xFF for i in range(16))
        plain = Snow2(key).decrypt(bytes(data[pos : pos + 12]))
        hash_, version, entry_count = struct.unpack_from("<iBi", plain, 0)
        if version != 2:
            raise ValueError(f"{self.path}: not an ms v1 file (version byte {version})")
        actual = hashed_salt_len + version + entry_count + sum(struct.unpack_from(f"<{salt_len}H", salt_bytes, 0))
        if _to_i32(actual) != hash_:
            raise ValueError(f"{self.path}: ms v1 header hash mismatch")
        entry_start = pos + 9 + sum(ord(c) * 3 for c in name) % 212 + 33

        n = len(nws)
        key2 = bytes((i + (i % 3 + 2) * ord(nws[n - 1 - i % n])) & 0xFF for i in range(16))
        cipher = Snow2(key2)
        buf = bytearray()
        src = entry_start
        cursor = 0

        def take(count: int) -> bytes:
            nonlocal src, cursor
            while len(buf) - cursor < count:
                chunk = bytes(data[src : src + 4096])
                chunk = chunk[: len(chunk) & ~3]
                if not chunk:
                    raise EOFError(f"{self.path}: truncated entry table")
                buf.extend(cipher.decrypt(chunk))
                src += len(chunk)
            out = bytes(buf[cursor : cursor + count])
            cursor += count
            return out

        raw = []
        for _ in range(entry_count):
            (name_len,) = struct.unpack("<i", take(4))
            entry_name = take(name_len * 2).decode("utf-16-le")
            checksum, _flags, start, size, size_aligned, _u1, _u2 = struct.unpack("<7i", take(28))
            raw.append((entry_name, checksum, start, size, size_aligned, take(16)))
        data_start = self._align_1k(entry_start + ((cursor + 3) & ~3))
        self.entries = [MsEntry(nm, cs, data_start + st * 1024, sz, sza, k) for nm, cs, st, sz, sza, k in raw]

    def read_entry(self, entry: MsEntry) -> bytes:
        """Decrypt and return the plain bytes of `entry`."""
        key = _entry_key(entry, _fnv1a(self.salt), 16)
        enc = bytes(self.data[entry.start : entry.start + entry.size_aligned])
        plain = bytearray(Snow2(key).decrypt(enc))
        head = min(entry.size, 1024) & ~3
        if head:
            words = np.frombuffer(bytes(plain[:head]), dtype="<u4")
            plain[:head] = (words - Snow2(key).keystream(len(words))).astype("<u4").tobytes()
        return bytes(plain[: entry.size])


class _ChaChaTableReader:
    """Replicates the reference `ChaCha20Reader`, including its counter-reset quirk.

    The reference decrypts 64-byte blocks and resets the block counter to 0 whenever a read call
    ends exactly at the end of a block, so callers must issue reads with the same granularity as
    the reference (one call per int32 field).
    """

    def __init__(self, data: mmap.mmap, start: int, key: bytes) -> None:
        self._data = data
        self._src = start
        self._key = key
        self._nonce = bytes(12)
        self._counter = 0
        self._block = b""
        self._used = 64

    @property
    def position(self) -> int:
        return self._src

    def read(self, count: int) -> bytes:
        out = bytearray()
        while count > 0:
            if self._used >= 64:
                enc = bytes(self._data[self._src : self._src + 64])
                if len(enc) < 64:
                    raise EOFError("truncated ms entry table")
                self._src += 64
                self._block = _chacha20(self._key, self._nonce, self._counter, enc)
                self._counter += 1
                self._used = 0
            take = min(count, 64 - self._used)
            out += self._block[self._used : self._used + take]
            self._used += take
            count -= take
        if self._used >= 64:
            self._counter = 0
        return bytes(out)

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]


class MsFileV2(MsFileBase):
    """`.ms` layout with header version 4 (ChaCha20, partial image encryption)."""

    def _read(self) -> None:
        name = self._file_name
        data = self.data
        rand_count = sum(ord(c) for c in name) % 312 + 30
        rand = [((b - 256 if b > 127 else b) >> 1) & 0xFF for b in data[:rand_count]]
        pos = rand_count
        encrypted_version = data[pos]
        pos += 1
        version = encrypted_version ^ rand[0]
        if version != 4:
            raise ValueError(f"{self.path}: not an ms v2 file (version {version})")
        (hashed_salt_len,) = struct.unpack_from("<i", data, pos)
        pos += 4
        salt_len = (hashed_salt_len & 0xFF) ^ rand[0]
        salt_bytes = bytes(data[pos : pos + salt_len * 2])
        pos += salt_len * 2
        chars = []
        for i in range(salt_len):
            a = rand[i] ^ salt_bytes[i * 2]
            chars.append(chr((((a | 0x4B) << 1) - a - 75) & 0xFFFF))
        self.salt = "".join(chars)
        nws = name + self.salt
        n = len(nws)

        key = bytes(((ord(nws[i % n]) + i) & 0xFF) ^ _CHACHA_KEY_OBSCURE[i] for i in range(32))
        plain = _chacha20(key, bytes(12), 0, bytes(data[pos : pos + 8]))
        hash_, entry_count = struct.unpack("<ii", plain)
        actual = (
            hashed_salt_len
            + encrypted_version
            + version
            + entry_count
            + sum(struct.unpack_from(f"<{salt_len}H", salt_bytes, 0))
        )
        if _to_i32(actual) != hash_:
            raise ValueError(f"{self.path}: ms v2 header hash mismatch")
        entry_start = pos + 8 + sum(ord(c) * 3 for c in name) % 212 + 64

        key2 = bytes(((i + (i % 3 + 2) * ord(nws[n - 1 - i % n])) & 0xFF) ^ _CHACHA_KEY_OBSCURE[i] for i in range(32))
        reader = _ChaChaTableReader(data, entry_start, key2)
        raw = []
        for _ in range(entry_count):
            # Read call granularity matters: the counter resets when a call ends on a block boundary.
            name_len = reader.i32()
            entry_name = reader.read(name_len * 2).decode("utf-16-le")
            checksum, _flags, start, size, size_aligned, _u1, _u2 = (reader.i32() for _ in range(7))
            entry_key = reader.read(16)
            reader.i32()  # unk3
            reader.i32()  # unk4
            raw.append((entry_name, checksum, start, size, size_aligned, entry_key))
        data_start = self._align_1k(reader.position)
        self.entries = [MsEntry(nm, cs, data_start + st * 1024, sz, sza, k) for nm, cs, st, sz, sza, k in raw]

    def read_entry(self, entry: MsEntry) -> bytes:
        """Decrypt and return the plain bytes of `entry`."""
        key_hash = _fnv1a(self.salt)
        key_hash2 = key_hash >> 1
        key_hash3 = key_hash2 ^ 0x6C
        img_key = bytes(b ^ o for b, o in zip(_entry_key(entry, key_hash, 32), _CHACHA_KEY_OBSCURE, strict=True))

        hash_data = bytearray(struct.pack("<III", key_hash, key_hash2, key_hash3))
        a = b = d = 0
        c = 90
        for i in range(12):
            hash_data[i] ^= (d + 11 * (i // 11) + (c ^ (i >> 2)) + (a ^ b)) & 0xFF
            d -= 1
            a += 8
            b += 17
            c += 43
        nonce = bytes(4) + bytes(hash_data[:8])
        (counter,) = struct.unpack_from("<I", hash_data, 8)

        head = min(entry.size, 1024)
        enc_head = bytes(self.data[entry.start : entry.start + head])
        plain_head = _chacha20(img_key, nonce, counter, enc_head)
        if entry.size <= 1024:
            return plain_head
        return plain_head + bytes(self.data[entry.start + 1024 : entry.start + entry.size])


def open_ms_file(path: Path, key_hint: KeyHint) -> MsFileBase:
    """Open an `.ms` pack, trying the v1 and then the v2 layout.

    Raises
    ------
    ValueError
        If neither layout matches.
    """
    errors = []
    for cls in (MsFileV1, MsFileV2):
        try:
            return cls(path, key_hint)
        except (ValueError, EOFError, struct.error, UnicodeDecodeError) as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise ValueError(f"Failed to open {path}: {'; '.join(errors)}")
