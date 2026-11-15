"""WZ string/data XOR keys and version hashing (port of `Wz_Crypto.cs` / `WzVersionHasher.cs`)."""

from __future__ import annotations

import enum
import functools

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

_AES_KEY = bytes(
    [
        0x13, 0, 0, 0, 0x08, 0, 0, 0, 0x06, 0, 0, 0, 0xB4, 0, 0, 0,
        0x1B, 0, 0, 0, 0x0F, 0, 0, 0, 0x33, 0, 0, 0, 0x52, 0, 0, 0,
    ]
)  # fmt: skip


class KeyType(enum.Enum):
    """Known PKG1 key families."""

    BMS = "bms"
    """No encryption (all-zero key)."""

    KMS = "kms"
    """Korean client IV."""

    GMS = "gms"
    """Global client IV."""


_IVS: dict[KeyType, bytes | None] = {
    KeyType.BMS: None,
    KeyType.KMS: bytes([0xB9, 0x7D, 0x63, 0xE9]),
    KeyType.GMS: bytes([0x4D, 0x23, 0xC7, 0x2B]),
}


class WzKey:
    """An XOR keystream, grown on demand, with precomputed string masks.

    The WZ string encodings XOR each byte (or UTF-16 code unit) with both the keystream and an
    incrementing mask. The combined masks are cached here so strings decode with one big-int XOR.
    """

    def __init__(self, key_type: KeyType) -> None:
        self.key_type = key_type
        self._iv = _IVS[key_type]
        self._keys = bytearray()
        self._ascii_mask = b""
        self._utf16_mask = b""

    def keystream(self, size: int) -> bytes:
        """Return the first `size` bytes of the keystream."""
        self._ensure(size)
        return bytes(self._keys[:size])

    def _ensure(self, size: int) -> None:
        if len(self._keys) >= size:
            return
        size = (size + 4095) & ~4095
        if self._iv is None:
            self._keys = bytearray(size)
            return
        encryptor = Cipher(algorithms.AES(_AES_KEY), modes.ECB()).encryptor()
        keys = self._keys
        block = bytes(self._iv[i % 4] for i in range(16)) if not keys else bytes(keys[-16:])
        while len(keys) < size:
            block = encryptor.update(block)
            keys += block
        self._keys = keys

    def xor(self, data: bytes, offset: int = 0) -> bytes:
        """XOR `data` with the keystream starting at `offset`."""
        if self._iv is None or not data:
            return bytes(data)
        n = len(data)
        self._ensure(offset + n)
        key = self._keys[offset : offset + n]
        return (int.from_bytes(data, "little") ^ int.from_bytes(key, "little")).to_bytes(n, "little")

    def decode_ascii(self, data: bytes) -> str:
        """Decode a masked single-byte WZ string."""
        n = len(data)
        if len(self._ascii_mask) < n:
            self._build_masks(n)
        mask = self._ascii_mask[:n]
        raw = (int.from_bytes(data, "little") ^ int.from_bytes(mask, "little")).to_bytes(n, "little")
        return raw.decode("latin-1")

    def decode_utf16(self, data: bytes, apply_mask: bool = True) -> str:
        """Decode a masked UTF-16LE WZ string."""
        n = len(data)
        if apply_mask:
            if len(self._utf16_mask) < n:
                self._build_masks(n // 2)
            mask = self._utf16_mask[:n]
        else:
            mask = self.keystream(n) if self._iv is not None else bytes(n)
        raw = (int.from_bytes(data, "little") ^ int.from_bytes(mask, "little")).to_bytes(n, "little")
        return raw.decode("utf-16-le", errors="replace")

    def _build_masks(self, chars: int) -> None:
        chars = max(chars, 256)
        chars = (chars + 255) & ~255
        self._ensure(chars * 2)
        keys = self._keys
        self._ascii_mask = bytes((keys[i] ^ (0xAA + i)) & 0xFF for i in range(chars))
        utf16 = bytearray(chars * 2)
        for i in range(chars):
            m = (keys[2 * i] | (keys[2 * i + 1] << 8)) ^ ((0xAAAA + i) & 0xFFFF)
            utf16[2 * i] = m & 0xFF
            utf16[2 * i + 1] = m >> 8
        self._utf16_mask = bytes(utf16)


_KEY_CACHE: dict[KeyType, WzKey] = {}


def get_key(key_type: KeyType) -> WzKey:
    """Return the shared `WzKey` for `key_type`."""
    key = _KEY_CACHE.get(key_type)
    if key is None:
        key = WzKey(key_type)
        _KEY_CACHE[key_type] = key
    return key


def all_keys() -> list[WzKey]:
    """All candidate keys in detection order."""
    return [get_key(k) for k in (KeyType.BMS, KeyType.GMS, KeyType.KMS)]


def pkg1_hash_version(version: int) -> int:
    """Compute the 32-bit version hash used for PKG1 offset decryption."""
    h = 0
    for c in str(version):
        h = ((h << 5) + ord(c) + 1) & 0xFFFFFFFF
    return h


def pkg1_encrypted_version(hash_version: int) -> int:
    """Compute the 1-byte encrypted version stored in PKG1 headers."""
    h = hash_version
    return 0xFF ^ ((h >> 24) & 0xFF) ^ ((h >> 16) & 0xFF) ^ ((h >> 8) & 0xFF) ^ (h & 0xFF)


@functools.cache
def pkg1_version_candidates(encrypted_version: int, limit: int = 0x7FFF) -> tuple[tuple[int, int], ...]:
    """List `(version, hash_version)` pairs whose encrypted version matches."""
    out = []
    for v in range(1, limit):
        h = pkg1_hash_version(v)
        if pkg1_encrypted_version(h) == encrypted_version:
            out.append((v, h))
    return tuple(out)


def pkg1_offset(file_pos: int, hashed_offset: int, header_size: int, hash_version: int) -> int:
    """Decrypt a PKG1 directory-entry data offset (port of `Pkg1OffsetCalc`)."""
    m = 0xFFFFFFFF
    off = (file_pos - header_size) & m
    off = ~off & m
    off = (off * hash_version) & m
    off = (off - 0x581C3F6D) & m
    d = off & 0x1F
    off = ((off << d) | (off >> (32 - d))) & m if d else off
    off ^= hashed_offset
    return (off + header_size * 2) & m
