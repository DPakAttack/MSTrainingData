"""Item indexing: where each droppable item lives, its strings, and its ground-drop sprite frames."""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

from wz.node import WzNode
from wz.structure import WzStructure
from wz.values import WzCanvas

DEFAULT_FRAME_DELAY = 120
"""Delay (ms) for frames without a `delay` value, matching WzComparerR2's `Frame` default."""

_ID_IMG_RE = re.compile(r"^(\d+)\.img$")


class ItemCategory(enum.Enum):
    """Item group; the value is the output folder name."""

    EQUIP = "Equip"
    CONSUME = "Consume"
    INSTALL = "Install"
    ETC = "Etc"
    CASH = "Cash"
    PET = "Pet"
    SPECIAL = "Special"


EQUIP_FOLDERS = (
    "Accessory",
    "Cap",
    "Cape",
    "Coat",
    "Glove",
    "Longcoat",
    "Pants",
    "PetEquip",
    "Ring",
    "Shield",
    "Shoes",
    "Weapon",
    "TamingMob",
)
"""`Character/` folders holding droppable equipment (faces, hair, and skins have no drop icon)."""

_GROUPED_CATEGORIES = (
    ItemCategory.CONSUME,
    ItemCategory.INSTALL,
    ItemCategory.ETC,
    ItemCategory.CASH,
    ItemCategory.SPECIAL,
)
_FLAT_STRING_IMAGES = ("Consume.img", "Ins.img", "Cash.img", "Pet.img")


@dataclass(frozen=True)
class ItemRef:
    """Location of one item."""

    item_id: int
    category: ItemCategory
    path: str
    """Path of the item node, e.g. `Item/Consume/0200.img/02000000`."""

    image_path: str
    """Path of the `.img` containing the item (equal to `path` for equipment and pets)."""


def index_items(wz: WzStructure) -> dict[int, ItemRef]:
    """Map every item id to its location.

    Equipment and pets are enumerated from `.img` file names without parsing them; grouped
    categories (`Item/Consume/0200.img`, ...) parse only each image's top-level directory.
    Non-numeric entries such as `_Canvas` or `MaplePoint.img` are skipped.
    """
    out: dict[int, ItemRef] = {}
    character = wz.find("Character")
    for folder_name in EQUIP_FOLDERS:
        folder = character.get(folder_name) if character is not None else None
        for img in folder or ():
            if m := _ID_IMG_RE.match(img.name):
                out[int(m.group(1))] = ItemRef(int(m.group(1)), ItemCategory.EQUIP, img.path, img.path)
    item = wz.find("Item")
    pets = item.get(ItemCategory.PET.value) if item is not None else None
    for img in pets or ():
        if m := _ID_IMG_RE.match(img.name):
            out[int(m.group(1))] = ItemRef(int(m.group(1)), ItemCategory.PET, img.path, img.path)
    for category in _GROUPED_CATEGORIES:
        folder = item.get(category.value) if item is not None else None
        for img in folder or ():
            if not _ID_IMG_RE.match(img.name):
                continue
            for entry in img:
                if entry.name.isdigit():
                    out[int(entry.name)] = ItemRef(int(entry.name), category, entry.path, img.path)
    return dict(sorted(out.items()))


def load_strings(wz: WzStructure) -> dict[int, tuple[str | None, str | None]]:
    """`(name, desc)` per item id from `String/`, following WzComparerR2's `StringLinker` layout."""
    out: dict[int, tuple[str | None, str | None]] = {}

    def add(entries: WzNode | None) -> None:
        for entry in entries or ():
            if entry.name.isdigit():
                target = entry.resolve()
                out[int(entry.name)] = (target.get_str("name"), target.get_str("desc"))

    for name in _FLAT_STRING_IMAGES:
        add(wz.find(f"String/{name}"))
    add(wz.find("String/Etc.img/Etc"))
    for region in wz.find("String/Eqp.img") or ():
        for folder in region:
            add(folder)
    return out


def load_names(wz: WzStructure) -> dict[int, str | None]:
    """Display name per item id (see `load_strings`)."""
    return {item_id: name for item_id, (name, _) in load_strings(wz).items()}


def drop_frames(item: WzNode) -> list[WzNode]:
    """Frames of the sprite drawn when the item lies on the ground.

    Uses `iconRaw` (falling back to `icon`) from `info/` or, for `Special` items, the item node
    itself. Animated sprites (meso) return their numbered frames in order; others return the
    single canvas node. Empty when the item has no icon.
    """
    for parent in (item.get("info"), item):
        if parent is None:
            continue
        for key in ("iconRaw", "icon"):
            node = parent.get(key)
            if node is None:
                continue
            node = node.resolve()
            if isinstance(node.value, WzCanvas):
                return [node]
            frames = sorted((c for c in node if c.name.isdigit()), key=lambda c: int(c.name))
            frames = [f for f in frames if isinstance(f.resolve().value, WzCanvas)]
            if frames:
                return frames
    return []


def inventory_icon(item: WzNode) -> WzNode | None:
    """The `icon` canvas node (inventory icon), or `None`."""
    for parent in (item.get("info"), item):
        node = parent.get("icon") if parent is not None else None
        if node is not None and isinstance(node.resolve().value, WzCanvas):
            return node.resolve()
    return None
