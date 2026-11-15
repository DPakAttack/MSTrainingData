"""Unit tests for crypto, binary reading, ms table decryption, and pixel decoding."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from wz.binary_reader import WzReader
from wz.canvas import TextureFormat, _decode_pixels, raw_size
from wz.crypto import (
    KeyType,
    get_key,
    pkg1_encrypted_version,
    pkg1_hash_version,
    pkg1_version_candidates,
)
from wz.ms_file import _chacha20, _ChaChaTableReader
from wz.snow2 import ALPHA_INV_MUL, ALPHA_MUL, T0, Snow2


def test_snow2_tables_match_reference() -> None:
    assert ALPHA_MUL[0] == 0
    assert ALPHA_MUL[1] == 0xE19FCF13
    assert ALPHA_INV_MUL[1] == 0x180F40CD
    assert T0[0] == 0xA56363C6


def test_snow2_keystream_is_deterministic() -> None:
    key = bytes(range(16))
    a = Snow2(key).keystream(32)
    b = Snow2(key).keystream(32)
    assert np.array_equal(a, b)
    assert len(set(a.tolist())) > 16


def test_pkg1_version_hash() -> None:
    assert pkg1_hash_version(83) == 1876
    assert pkg1_encrypted_version(pkg1_hash_version(83)) == 0xAC
    candidates = pkg1_version_candidates(0xAC)
    assert (83, 1876) in candidates
    assert all(pkg1_encrypted_version(h) == 0xAC and pkg1_hash_version(v) == h for v, h in candidates)


def _encode_ascii(text: str) -> bytes:
    return bytes(ord(c) ^ ((0xAA + i) & 0xFF) for i, c in enumerate(text))


def _encode_utf16(text: str) -> bytes:
    return b"".join(struct.pack("<H", ord(c) ^ ((0xAAAA + i) & 0xFFFF)) for i, c in enumerate(text))


def test_bms_key_string_decoding() -> None:
    key = get_key(KeyType.BMS)
    assert key.keystream(16) == bytes(16)
    assert key.decode_ascii(_encode_ascii("Property")) == "Property"
    assert key.decode_utf16(_encode_utf16("달팽이")) == "달팽이"


def test_reader_compressed_ints_and_strings() -> None:
    key = get_key(KeyType.BMS)
    name = _encode_ascii("origin")
    data = (
        bytes([5])
        + bytes([0x80])
        + struct.pack("<i", 123456)
        + bytes([0x00, 0x100 - len(name)])
        + name
        + bytes([0x01])
        + struct.pack("<i", 7)
    )
    r = WzReader(data)
    assert r.cint() == 5
    assert r.cint() == 123456
    assert r.image_string(key) == "origin"
    assert r.image_string(key) == "origin"
    assert r.pos == len(data)


def test_chacha_table_reader_resets_counter_on_block_boundary() -> None:
    key = bytes(range(32))
    plain = struct.pack("<32i", *range(32))
    nonce = bytes(12)
    cipher = _chacha20(key, nonce, 0, plain[:64]) + _chacha20(key, nonce, 0, plain[64:])
    reader = _ChaChaTableReader(cipher, 0, key)
    assert [reader.i32() for _ in range(32)] == list(range(32))


def test_chacha_table_reader_continues_counter_within_a_call() -> None:
    key = bytes(range(32))
    plain = bytes(range(128))
    nonce = bytes(12)
    cipher = _chacha20(key, nonce, 0, plain[:64]) + _chacha20(key, nonce, 1, plain[64:])
    assert _ChaChaTableReader(cipher, 0, key).read(100) == plain[:100]


@pytest.mark.parametrize(
    ("fmt", "w", "h", "expected"),
    [
        (TextureFormat.ARGB4444, 3, 2, 12),
        (TextureFormat.ARGB8888, 3, 2, 24),
        (TextureFormat.DXT1, 5, 5, 32),
        (TextureFormat.DXT5, 4, 4, 16),
    ],
)
def test_raw_size(fmt: int, w: int, h: int, expected: int) -> None:
    assert raw_size(fmt, w, h) == expected


def test_decode_argb4444() -> None:
    raw = struct.pack("<2H", 0xF12F, 0x0000)
    img = _decode_pixels(TextureFormat.ARGB4444, raw, 2, 1)
    assert img.getpixel((0, 0)) == (0x11, 0x22, 0xFF, 0xFF)
    assert img.getpixel((1, 0)) == (0, 0, 0, 0)


def test_decode_argb8888_and_rgb565() -> None:
    img = _decode_pixels(TextureFormat.ARGB8888, bytes([1, 2, 3, 4]), 1, 1)
    assert img.getpixel((0, 0)) == (3, 2, 1, 4)
    img = _decode_pixels(TextureFormat.RGB565, struct.pack("<H", 0xF800), 1, 1)
    assert img.getpixel((0, 0)) == (255, 0, 0, 255)
