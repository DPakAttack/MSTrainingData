"""Equipment slots of a character and how item ids map onto them."""

from __future__ import annotations

import enum
from dataclasses import dataclass

SKIN_COUNT = 7
"""Skin tones: body `000020NN.img` and head `000120NN.img` for NN in 0..6."""


class EquipType(enum.Enum):
    """A character slot. The declaration order is the codec's bit order; never reorder."""

    SKIN = "skin"
    FACE = "face"
    HAIR = "hair"
    CAP = "cap"
    FACE_ACCESSORY = "face_accessory"
    EYE_ACCESSORY = "eye_accessory"
    EARRINGS = "earrings"
    COAT = "coat"
    LONGCOAT = "longcoat"
    PANTS = "pants"
    SHOES = "shoes"
    GLOVE = "glove"
    CAPE = "cape"
    SHIELD = "shield"
    WEAPON = "weapon"


@dataclass(frozen=True)
class SlotInfo:
    """Static description of one slot."""

    folder: str
    """Folder under `Character/` holding the slot's `.img` files (`""` for the root)."""

    prefixes: tuple[int, ...]
    """Accepted values of `id // 10000`."""

    base: int
    """Subtracted from ids by the codec to keep the encoding short."""

    required: bool = False
    """Whether every character must have this slot."""


SLOTS: dict[EquipType, SlotInfo] = {
    EquipType.SKIN: SlotInfo("", (), 0, required=True),
    EquipType.FACE: SlotInfo("Face", (2,), 20000, required=True),
    EquipType.HAIR: SlotInfo("Hair", (3, 4, 6), 30000, required=True),
    EquipType.CAP: SlotInfo("Cap", (100,), 1000000),
    EquipType.FACE_ACCESSORY: SlotInfo("Accessory", (101,), 1010000),
    EquipType.EYE_ACCESSORY: SlotInfo("Accessory", (102,), 1020000),
    EquipType.EARRINGS: SlotInfo("Accessory", (103,), 1030000),
    EquipType.COAT: SlotInfo("Coat", (104,), 1040000),
    EquipType.LONGCOAT: SlotInfo("Longcoat", (105,), 1050000),
    EquipType.PANTS: SlotInfo("Pants", (106,), 1060000),
    EquipType.SHOES: SlotInfo("Shoes", (107,), 1070000),
    EquipType.GLOVE: SlotInfo("Glove", (108,), 1080000),
    EquipType.CAPE: SlotInfo("Cape", (110,), 1100000),
    EquipType.SHIELD: SlotInfo("Shield", (109,), 1090000),
    EquipType.WEAPON: SlotInfo("Weapon", (*range(121, 160), 170), 1210000),
}

REQUIRED_SLOTS = tuple(t for t, s in SLOTS.items() if s.required)
OPTIONAL_SLOTS = tuple(t for t, s in SLOTS.items() if not s.required)

VSLOT_PRECEDENCE = (
    EquipType.SKIN,
    EquipType.FACE,
    EquipType.HAIR,
    EquipType.COAT,
    EquipType.LONGCOAT,
    EquipType.PANTS,
    EquipType.SHOES,
    EquipType.GLOVE,
    EquipType.CAPE,
    EquipType.SHIELD,
    EquipType.WEAPON,
    EquipType.EARRINGS,
    EquipType.FACE_ACCESSORY,
    EquipType.EYE_ACCESSORY,
    EquipType.CAP,
)
"""Order in which parts claim their `vslot` codes; later parts win (equipment covers the body)."""

GENDER_MALE = 0
GENDER_FEMALE = 1
GENDER_UNISEX = 2


def equip_type_of(item_id: int) -> EquipType:
    """Infer the slot of an item id (skin tones are not ids; use `EquipType.SKIN` explicitly).

    Raises
    ------
    ValueError
        If the id does not belong to a supported slot.
    """
    prefix = item_id // 10000
    for equip_type, slot in SLOTS.items():
        if prefix in slot.prefixes:
            return equip_type
    raise ValueError(f"Item {item_id} does not belong to a supported character slot")


def item_gender(item_id: int) -> int:
    """`GENDER_MALE`, `GENDER_FEMALE`, or `GENDER_UNISEX` from the id's thousands digit."""
    digit = item_id // 1000 % 10
    return digit if digit in (GENDER_MALE, GENDER_FEMALE) else GENDER_UNISEX


def hair_style(hair_id: int) -> int:
    """Hair style key: ids sharing it differ only in color (the last digit)."""
    return hair_id // 10


def face_style(face_id: int) -> int:
    """Face style key: ids sharing it differ only in color (the hundreds digit)."""
    return face_id - (face_id // 100 % 10) * 100


def is_cash_weapon(item_id: int) -> bool:
    """Cash weapons (`170xxxx`) store frames under a weapon-type key."""
    return item_id // 10000 == 170
