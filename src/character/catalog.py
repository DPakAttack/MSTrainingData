"""Index of drawable character items and the metadata needed to combine them."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from character.equip import SKIN_COUNT, SLOTS, EquipType, equip_type_of, is_cash_weapon, item_gender
from wz.node import WzNode
from wz.structure import WzStructure

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1
_TWO_HANDED_WEAPON_TYPES = frozenset(range(40, 47))


def split_slot_codes(codes: str | None) -> tuple[str, ...]:
    """Split a concatenated slot string such as `"MaPn"` into `("Ma", "Pn")`."""
    if not codes:
        return ()
    return tuple(codes[i : i + 2] for i in range(0, len(codes) - len(codes) % 2, 2))


@dataclass(frozen=True)
class ItemInfo:
    """What the builder and renderer need to know about one item."""

    item_id: int
    """Item id (the skin tone index for `EquipType.SKIN`)."""

    slot: EquipType
    """Slot the item goes into."""

    islot: tuple[str, ...]
    """Slot codes the item occupies; items sharing a code cannot be worn together."""

    vslot: tuple[str, ...]
    """Slot codes whose sprite layers this item takes over (used for visibility)."""

    gender: int
    """`GENDER_MALE`, `GENDER_FEMALE`, or `GENDER_UNISEX`."""

    two_handed: bool = False
    """Weapons only: the character uses the `stand2` pose."""

    weapon_type: str | None = None
    """Cash weapons only: the weapon-type key whose frames are drawn."""

    invisible_face: bool = False
    """Face accessories only: hides the face while worn."""

    @property
    def path(self) -> str:
        """Path of the item's `.img` in the data tree."""
        if self.slot == EquipType.SKIN:
            return f"Character/{2000 + self.item_id:08d}.img"
        return f"Character/{SLOTS[self.slot].folder}/{self.item_id:08d}.img"


def _has_frames(node: WzNode) -> bool:
    return any(node.get(k) is not None for k in ("stand1", "stand2", "default"))


def _item_info(node: WzNode, item_id: int, slot: EquipType) -> ItemInfo | None:
    weapon_type = None
    two_handed = False
    if slot == EquipType.WEAPON and is_cash_weapon(item_id):
        keys = sorted((k.name for k in node if k.name.isdigit() and _has_frames(k)), key=int)
        if not keys:
            return None
        weapon_type = keys[0]
        two_handed = int(weapon_type) in _TWO_HANDED_WEAPON_TYPES
    elif not _has_frames(node):
        return None
    elif slot == EquipType.WEAPON:
        two_handed = node.get_int("info/stand") == 2
    return ItemInfo(
        item_id=item_id,
        slot=slot,
        islot=split_slot_codes(node.get_str("info/islot")),
        vslot=split_slot_codes(node.get_str("info/vslot")),
        gender=item_gender(item_id),
        two_handed=two_handed,
        weapon_type=weapon_type,
        invisible_face=bool(node.get_int("info/invisibleFace")),
    )


class CharacterCatalog:
    """Lazily built index of every drawable item per slot.

    Parameters
    ----------
    wz
        Mounted data tree.
    cache_path
        Optional JSON file used to persist the index between runs. It is rebuilt when the
        Character package files change (by name and size).
    """

    def __init__(self, wz: WzStructure, cache_path: Path | None = None) -> None:
        self.wz = wz
        self.cache_path = cache_path
        self._items: dict[int, ItemInfo] | None = None
        self._skins: dict[int, ItemInfo] = {}

    def items(self) -> dict[int, ItemInfo]:
        """All indexed items (excluding skins) by id."""
        if self._items is None:
            self._items = self._load_cache() or self._build()
        return self._items

    def ids(self, slot: EquipType) -> list[int]:
        """Valid ids for `slot`, ascending (skin tone indices for `EquipType.SKIN`)."""
        if slot == EquipType.SKIN:
            return [i for i in range(SKIN_COUNT) if self.wz.find(f"Character/{2000 + i:08d}.img") is not None]
        return sorted(i for i, info in self.items().items() if info.slot == slot)

    def get(self, slot: EquipType, item_id: int) -> ItemInfo | None:
        """Metadata for `item_id` in `slot`, or `None` if it is not a drawable item."""
        if slot == EquipType.SKIN:
            if item_id not in self._skins:
                node = self.wz.find(f"Character/{2000 + item_id:08d}.img")
                self._skins[item_id] = (
                    None
                    if node is None
                    else ItemInfo(item_id, slot, split_slot_codes(node.get_str("info/islot")), ("Bd",), 2)
                )
            return self._skins[item_id]
        info = self.items().get(item_id)
        return info if info is not None and info.slot == slot else None

    # Building -----------------------------------------------------------------------------

    def _fingerprint(self) -> list[list[object]]:
        folder = self.wz.root_dir / "Character"
        if not folder.is_dir():
            return []
        return sorted([p.relative_to(folder).as_posix(), p.stat().st_size] for p in folder.rglob("*.wz"))

    def _build(self) -> dict[int, ItemInfo]:
        start = time.monotonic()
        items: dict[int, ItemInfo] = {}
        for folder in sorted({s.folder for s in SLOTS.values() if s.folder}):
            parent = self.wz.find(f"Character/{folder}")
            for node in parent or ():
                stem = node.name.removesuffix(".img")
                if not node.name.endswith(".img") or not stem.isdigit():
                    continue
                item_id = int(stem)
                try:
                    slot = equip_type_of(item_id)
                    if SLOTS[slot].folder != folder:
                        continue
                    info = _item_info(node, item_id, slot)
                except (ValueError, OSError, KeyError) as exc:
                    logger.warning("Skipping %s: %s", node.path, exc)
                    continue
                finally:
                    node.unload()
                if info is not None:
                    items[item_id] = info
        logger.info("Indexed %d character items in %.1fs", len(items), time.monotonic() - start)
        self._save_cache(items)
        return items

    def _load_cache(self) -> dict[int, ItemInfo] | None:
        if self.cache_path is None or not self.cache_path.is_file():
            return None
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if data.get("version") != _CACHE_VERSION or data.get("fingerprint") != self._fingerprint():
                return None
            items = {}
            for raw in data["items"]:
                raw["slot"] = EquipType(raw["slot"])
                raw["islot"] = tuple(raw["islot"])
                raw["vslot"] = tuple(raw["vslot"])
                items[raw["item_id"]] = ItemInfo(**raw)
            return items
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring unreadable catalog cache %s: %s", self.cache_path, exc)
            return None

    def _save_cache(self, items: dict[int, ItemInfo]) -> None:
        if self.cache_path is None:
            return
        rows = []
        for item_id in sorted(items):
            row = asdict(items[item_id])
            row["slot"] = row["slot"].value
            rows.append(row)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": _CACHE_VERSION, "fingerprint": self._fingerprint(), "items": rows}
        self.cache_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
