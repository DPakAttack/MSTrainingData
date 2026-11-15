"""Pure-Python SNOW 2.0 stream cipher, decrypt mode (port of `Snow2CryptoTransform.cs`).

The `.ms` packs "decrypt" by subtracting keystream words from little-endian uint32 data words.
Lookup tables are generated from the AES S-box and the SNOW 2.0 field definitions at import time.
"""

from __future__ import annotations

import numpy as np

_M = 0xFFFFFFFF


def _aes_sbox() -> list[int]:
    sbox = [0] * 256
    p = q = 1
    while True:
        p = p ^ ((p << 1) & 0xFF) ^ (0x1B if p & 0x80 else 0)
        q ^= q << 1
        q ^= q << 2
        q ^= q << 4
        q &= 0xFF
        if q & 0x80:
            q ^= 0x09
        x = q ^ ((q << 1 | q >> 7) & 0xFF) ^ ((q << 2 | q >> 6) & 0xFF) ^ ((q << 3 | q >> 5) & 0xFF)
        x ^= (q << 4 | q >> 4) & 0xFF
        sbox[p] = x ^ 0x63
        if p == 1:
            break
    sbox[0] = 0x63
    return sbox


def _mulx(v: int, c: int) -> int:
    return ((v << 1) ^ c) & 0xFF if v & 0x80 else v << 1


def _mulx_pow(v: int, i: int, c: int) -> int:
    for _ in range(i):
        v = _mulx(v, c)
    return v


def _build_tables() -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    sbox = _aes_sbox()
    t0, t1, t2, t3 = [], [], [], []
    for x in range(256):
        s = sbox[x]
        s2 = _mulx(s, 0x1B)
        s3 = s2 ^ s
        w = (s3 << 24) | (s << 16) | (s << 8) | s2
        t0.append(w)
        t1.append(((w << 8) | (w >> 24)) & _M)
        t2.append(((w << 16) | (w >> 16)) & _M)
        t3.append(((w << 24) | (w >> 8)) & _M)
    a_mul = [
        (_mulx_pow(c, 23, 0xA9) << 24)
        | (_mulx_pow(c, 245, 0xA9) << 16)
        | (_mulx_pow(c, 48, 0xA9) << 8)
        | _mulx_pow(c, 239, 0xA9)
        for c in range(256)
    ]
    ainv_mul = [
        (_mulx_pow(c, 16, 0xA9) << 24)
        | (_mulx_pow(c, 39, 0xA9) << 16)
        | (_mulx_pow(c, 6, 0xA9) << 8)
        | _mulx_pow(c, 64, 0xA9)
        for c in range(256)
    ]
    return t0, t1, t2, t3, a_mul, ainv_mul


T0, T1, T2, T3, ALPHA_MUL, ALPHA_INV_MUL = _build_tables()


class Snow2:
    """Stateful SNOW 2.0 keystream with a 128-bit key and no IV.

    Parameters
    ----------
    key
        16-byte key. Bytes are sign-extended when packed into words, matching the reference.
    """

    def __init__(self, key: bytes) -> None:
        if len(key) != 16:
            raise ValueError("Snow2 key must be 16 bytes")
        sk = [b - 256 if b > 127 else b for b in key]
        k = [((sk[4 * i] << 24) | (sk[4 * i + 1] << 16) | (sk[4 * i + 2] << 8) | sk[4 * i + 3]) & _M for i in range(4)]
        s = [0] * 16
        s[15], s[14], s[13], s[12] = k
        s[11], s[10], s[9], s[8] = (~k[0] & _M, ~k[1] & _M, ~k[2] & _M, ~k[3] & _M)
        s[7], s[6], s[5], s[4] = k
        s[3], s[2], s[1], s[0] = (~k[0] & _M, ~k[1] & _M, ~k[2] & _M, ~k[3] & _M)
        r1 = r2 = 0
        am, aim = ALPHA_MUL, ALPHA_INV_MUL
        for _ in range(2):
            for i in range(16):
                out = ((r1 + s[(i + 15) & 15]) & _M) ^ r2
                v = s[i]
                w = s[(i + 11) & 15]
                s[i] = ((v << 8) & _M) ^ am[v >> 24] ^ s[(i + 2) & 15] ^ (w >> 8) ^ aim[w & 0xFF] ^ out
                tmp = (r2 + s[(i + 5) & 15]) & _M
                r2 = T0[r1 & 0xFF] ^ T1[(r1 >> 8) & 0xFF] ^ T2[(r1 >> 16) & 0xFF] ^ T3[r1 >> 24]
                r1 = tmp
        self._s = s
        self._r1 = r1
        self._r2 = r2
        self._pending: list[int] = []

    def _block(self) -> list[int]:
        s = self._s
        r1, r2 = self._r1, self._r2
        am, aim = ALPHA_MUL, ALPHA_INV_MUL
        t0, t1, t2, t3 = T0, T1, T2, T3
        ks = [0] * 16
        for i in range(16):
            v = s[i]
            w = s[(i + 11) & 15]
            v = ((v << 8) & _M) ^ am[v >> 24] ^ s[(i + 2) & 15] ^ (w >> 8) ^ aim[w & 0xFF]
            s[i] = v
            tmp = (r2 + s[(i + 5) & 15]) & _M
            r2 = t0[r1 & 0xFF] ^ t1[(r1 >> 8) & 0xFF] ^ t2[(r1 >> 16) & 0xFF] ^ t3[r1 >> 24]
            r1 = tmp
            ks[i] = (((r1 + v) & _M) ^ r2) ^ s[(i + 1) & 15]
        self._r1, self._r2 = r1, r2
        return ks

    def keystream(self, words: int) -> np.ndarray:
        """Return the next `words` keystream words."""
        out = self._pending
        while len(out) < words:
            out.extend(self._block())
        result, self._pending = out[:words], out[words:]
        return np.array(result, dtype=np.uint32)

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt `data`, continuing the stream. Trailing bytes beyond a whole word are dropped."""
        n = len(data) // 4
        if n == 0:
            return b""
        words = np.frombuffer(data, dtype="<u4", count=n)
        return (words - self.keystream(n)).astype("<u4").tobytes()


def snow2_decrypt(key: bytes, data: bytes) -> bytes:
    """Decrypt `data` with a fresh SNOW 2.0 stream keyed by `key`."""
    return Snow2(key).decrypt(data)
