"""`CharacterSpec` and its reversible, filesystem-safe name encoding.

Layout of the encoded bytes (before base32):

- 1 byte: format version (`1`).
- 2 bytes, little endian: bitmask of present slots, bit `i` = `list(EquipType)[i]`.
- For each present slot in `EquipType` order: unsigned LEB128 varint of `id - SLOTS[slot].base`.

The bytes are written as lowercase base32 without padding and prefixed with `c<version>_`,
e.g. `c1_aeaaaaaa...`. Only `[a-z2-7_]` is used, so names are safe on case-insensitive and
Windows filesystems.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import PurePath

from character.equip import REQUIRED_SLOTS, SLOTS, EquipType

VERSION = 1
_PREFIX = f"c{VERSION}_"
_ORDER = tuple(EquipType)


@dataclass(frozen=True)
class CharacterSpec(Mapping[EquipType, int]):
    """An immutable, conflict-free set of slot assignments (`EquipType.SKIN` holds a tone index).

    Iteration follows `EquipType` declaration order, so equal specs compare and encode equally.
    Use `CharacterApi.build` to create one from possibly conflicting assignments.
    """

    parts: tuple[tuple[EquipType, int], ...]
    """`(slot, id)` pairs in `EquipType` order."""

    @classmethod
    def from_mapping(cls, parts: Mapping[EquipType, int]) -> CharacterSpec:
        """Create a spec from `slot -> id`, ordering the slots canonically."""
        return cls(tuple((t, int(parts[t])) for t in _ORDER if t in parts))

    def __getitem__(self, key: EquipType) -> int:
        for slot, item_id in self.parts:
            if slot == key:
                return item_id
        raise KeyError(key)

    def __iter__(self) -> Iterator[EquipType]:
        return (slot for slot, _ in self.parts)

    def __len__(self) -> int:
        return len(self.parts)

    def missing_required(self) -> list[EquipType]:
        """Required slots (skin, face, hair) that are not assigned."""
        return [t for t in REQUIRED_SLOTS if t not in self]

    def to_dict(self) -> dict[str, int]:
        """`{slot name: id}`, e.g. for JSON."""
        return {slot.value: item_id for slot, item_id in self.parts}

    def encode(self) -> str:
        """The reversible name of this spec (see the module docstring)."""
        return encode(self)


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def encode(spec: CharacterSpec) -> str:
    """Encode `spec` as `c1_<base32>`.

    Raises
    ------
    ValueError
        If an id is below its slot's base (i.e. not a valid id for that slot).
    """
    mask = 0
    body = bytearray()
    for slot, item_id in spec.parts:
        offset = item_id - SLOTS[slot].base
        if offset < 0:
            raise ValueError(f"{item_id} is not a valid {slot.value} id")
        mask |= 1 << _ORDER.index(slot)
        body += _varint(offset)
    raw = bytes([VERSION]) + mask.to_bytes(2, "little") + bytes(body)
    return _PREFIX + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def decode(name: str | PurePath) -> CharacterSpec:
    """Invert `encode`. Accepts the bare name, a filename (`c1_....png`), or a path.

    Raises
    ------
    ValueError
        If the name is not a valid encoding.
    """
    stem = PurePath(name).name.split(".")[0]
    if not stem.startswith(_PREFIX):
        raise ValueError(f"Not a character name: {name}")
    text = stem[len(_PREFIX) :].upper()
    try:
        raw = base64.b32decode(text + "=" * (-len(text) % 8))
    except ValueError as exc:
        raise ValueError(f"Not a character name: {name}") from exc
    if len(raw) < 3 or raw[0] != VERSION:
        raise ValueError(f"Unsupported character name version in {name}")
    mask = int.from_bytes(raw[1:3], "little")
    pos = 3
    parts = []
    for i, slot in enumerate(_ORDER):
        if not mask >> i & 1:
            continue
        value = shift = 0
        while True:
            if pos >= len(raw):
                raise ValueError(f"Truncated character name: {name}")
            byte = raw[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        parts.append((slot, value + SLOTS[slot].base))
    if pos != len(raw) or mask >> len(_ORDER):
        raise ValueError(f"Malformed character name: {name}")
    return CharacterSpec(tuple(parts))
