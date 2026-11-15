"""Deck cards for mobs, items, and maps, and decoding a card into a positioned sprite."""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from PIL import Image

import items
from wz.node import WzNode
from wz.structure import WzStructure
from wz.values import WzCanvas, WzVector

logger = logging.getLogger(__name__)

MESO_IDS = (9000000, 9000001, 9000002, 9000003)
"""Meso drop sprites (bronze coin, gold coin, bills, bag), the money drops used for `is_meso`."""

_MOB_IMG_RE = re.compile(r"^(\d{7})\.img$")
_TRAILING_DIGITS_RE = re.compile(r"\d+$")


@dataclass(frozen=True)
class MobCard:
    """One animation frame of one mob."""

    mob_id: int
    action: str
    frame: int


@dataclass(frozen=True)
class ItemCard:
    """One drop-sprite frame of one item."""

    item_id: int
    frame: int


@dataclass(frozen=True)
class EntitySprite:
    """A decoded RGBA bitmap and its anchor (the foot point for mobs and drops)."""

    image: Image.Image
    origin: tuple[int, int]


def mob_state(action: str) -> str:
    """Action name without its variant number: `hit1` -> `hit`, `attack2` -> `attack`."""
    return _TRAILING_DIGITS_RE.sub("", action) or action


def _is_canvas(node: WzNode) -> bool:
    return isinstance(node.resolve().resolve_canvas().value, WzCanvas)


def _origin(node: WzNode) -> tuple[int, int]:
    value = node.get_value("origin")
    return (value.x, value.y) if isinstance(value, WzVector) else (0, 0)


class SpriteSource:
    """Builds deck cards and decodes cards into sprites, with an LRU cache of decoded bitmaps.

    Parameters
    ----------
    wz
        Mounted data tree.
    item_index
        `ItemApi.index` (item id -> location).
    cache_size
        Maximum decoded sprites kept in memory.
    """

    def __init__(self, wz: WzStructure, item_index: dict[int, items.ItemRef], cache_size: int = 4096) -> None:
        self._wz = wz
        self._items = item_index
        self._cache: OrderedDict[object, EntitySprite | None] = OrderedDict()
        self._cache_size = cache_size
        self._mob_links: dict[int, int] = {}

    # Cards ------------------------------------------------------------------------------

    def mob_ids(self) -> list[int]:
        mob = self._wz.find("Mob")
        return sorted(int(m.group(1)) for c in mob or () if (m := _MOB_IMG_RE.match(c.name)))

    def mob_node(self, mob_id: int) -> WzNode | None:
        return self._wz.find(f"Mob/{mob_id:07d}.img")

    def mob_source_id(self, mob_id: int) -> int:
        """The mob whose frames `mob_id` uses (`info/link`, or itself)."""
        if mob_id not in self._mob_links:
            node = self.mob_node(mob_id)
            link = node.get_int("info/link", 0) if node is not None else 0
            self._mob_links[mob_id] = link if link and self.mob_node(link) is not None else mob_id
        return self._mob_links[mob_id]

    def mob_cards(self) -> list[MobCard]:
        """Every numbered frame of every action of every mob (`info/link` followed)."""
        cards: list[MobCard] = []
        for mob_id in self.mob_ids():
            source = self.mob_node(self.mob_source_id(mob_id))
            if source is None:
                continue
            for action in source:
                if action.name == "info":
                    continue
                for frame in action:
                    if frame.name.isdigit() and _is_canvas(frame):
                        cards.append(MobCard(mob_id, action.name, int(frame.name)))
            source.unload()
        return cards

    def item_cards(self, ids: Iterable[int]) -> list[ItemCard]:
        """One card per drop-sprite frame of each item.

        Equipment drop sprites are single frames, so equipment is counted without loading its
        (large) images; other categories are inspected.
        """
        cards: list[ItemCard] = []
        loaded: set[str] = set()
        for item_id in ids:
            ref = self._items.get(item_id)
            if ref is None:
                continue
            if ref.category == items.ItemCategory.EQUIP:
                cards.append(ItemCard(item_id, 0))
                continue
            node = self._wz.find(ref.path)
            loaded.add(ref.image_path)
            count = len(items.drop_frames(node)) if node is not None else 0
            cards += [ItemCard(item_id, f) for f in range(count)]
        for path in loaded:
            image = self._wz.find(path)
            if image is not None:
                image.unload()
        return cards

    def meso_cards(self) -> list[ItemCard]:
        return self.item_cards(MESO_IDS)

    def drop_cards(self) -> list[ItemCard]:
        """Frames of every droppable item except the meso drops."""
        return self.item_cards(i for i in self._items if i not in MESO_IDS)

    # Sprites ----------------------------------------------------------------------------

    def _cached(self, key: object) -> tuple[bool, EntitySprite | None]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return True, self._cache[key]
        return False, None

    def _store(self, key: object, sprite: EntitySprite | None) -> EntitySprite | None:
        self._cache[key] = sprite
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return sprite

    def mob_sprite(self, card: MobCard) -> EntitySprite | None:
        """The frame bitmap of `card`, or `None` if it cannot be decoded."""
        hit, sprite = self._cached(card)
        if hit:
            return sprite
        source = self.mob_node(self.mob_source_id(card.mob_id))
        frame = source.find(f"{card.action}/{card.frame}") if source is not None else None
        sprite = None
        if frame is not None:
            frame = frame.resolve()
            try:
                sprite = EntitySprite(frame.resolve_canvas().value.to_image(), _origin(frame))
            except (AttributeError, ValueError, OSError) as exc:
                logger.warning("Cannot decode mob frame %s: %s", frame.path, exc)
        return self._store(card, sprite)

    def item_sprite(self, card: ItemCard) -> EntitySprite | None:
        """The drop-sprite frame of `card`, or `None` if it cannot be decoded."""
        hit, sprite = self._cached(card)
        if hit:
            return sprite
        ref = self._items.get(card.item_id)
        node = self._wz.find(ref.path) if ref is not None else None
        sprite = None
        frames = items.drop_frames(node) if node is not None else []
        if frames:
            frame = frames[card.frame % len(frames)]
            try:
                sprite = EntitySprite(frame.resolve_canvas().value.to_image(), _origin(frame))
            except (AttributeError, ValueError, OSError) as exc:
                logger.warning("Cannot decode item frame %s: %s", frame.path, exc)
        if ref is not None and ref.category == items.ItemCategory.EQUIP and node is not None:
            node.unload()
        return self._store(card, sprite)
