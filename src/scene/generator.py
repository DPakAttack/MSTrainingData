"""Synthetic scene generation: characters, mobs, and item drops standing on map platforms.

Output format
=============
`generate` writes one run folder, `<output_root>/Scenes/<YYYYmmdd-HHMMSS>[_N]/`, containing:

- `config.json`: the resolved config (including the seed actually used); passing it back to
  `generate_scene` reproduces the run.
- Per sample, with `<sample_id>` = `<index:06d>_<8 hex digits>`:
  - `<sample_id>.jpg`: the scene (RGB JPEG, quality 95).
  - `<sample_id>.bbox.jpg`: the same scene with each entity's `bbox` drawn (characters green,
    mobs red, items yellow, mesos orange) and a short label.
  - `<sample_id>.metadata.json`: the labels described below.

Coordinates and boxes
---------------------
All pixel coordinates are in the output image: origin at the top-left, x to the right, y down.
Every box is `[x_min, y_min, x_max, y_max]` with `x_max` / `y_max` EXCLUSIVE (so the width is
`x_max - x_min`). Boxes only cover non-transparent pixels (alpha >= 1); transparent sprite margins
never count.

- `bbox`: box of the entity's pixels that are VISIBLE in the final image, or `null` when none are.
  Entities are drawn on the map layer of the platform they stand on, so map scenery on higher
  layers, front backgrounds, portals, and entities drawn later can hide parts of them. A pixel
  counts as hidden when something drawn later covers it with alpha >= 128 (scenery) or >= 1
  (another entity).
- `full_bbox`: box of all of the sprite's non-transparent pixels, ignoring occlusion, clipped to
  the image (`null` if the sprite is entirely off-screen).
- `visible_fraction`: visible non-transparent pixels / all non-transparent pixels of the sprite
  (off-screen pixels count as not visible).
- `truncated`: whether the sprite's non-transparent pixels extend past the image edge.

metadata.json
-------------
{
  "sample_id": "000000_1a2b3c4d",
  "image": "000000_1a2b3c4d.jpg",
  "bbox_image": "000000_1a2b3c4d.bbox.jpg",
  "resolution": [1366, 768],                 # [width, height]
  "seed": 7,                                 # run seed (see config.json)
  "map": {
    "id": 100000000, "name": "Victoria Road - Henesys",   # name = "<street> - <map>"
    "street_name": "Victoria Road", "map_name": "Henesys",
    "viewport": [left, top, right, bottom]   # world rectangle shown (right/bottom exclusive)
  },
  "characters": [ENTITY + {
    "action": "stand1",                      # pose drawn (stand1, or stand2 with two-handed weapons)
    "spec": {"skin": 0, "face": 20000, ...}, # slot -> item id
    "code": "c1_..."                         # reversible name (character.codec.decode)
  }],
  "mobs": [ENTITY + {
    "mob_id": 100100, "mob_name": "Snail",
    "source_mob_id": 100100,                 # mob whose sprites were used (info/link target)
    "action": "hit1",                        # raw animation name
    "state": "hit",                          # action without its variant number
    "frame": 0                               # frame index within the action
  }],
  "items": [ENTITY + {
    "item_id": 2000000, "item_name": "Red Potion",   # name may be null
    "is_meso": false,                        # true for the meso drops 09000000-09000003
    "category": "Consume",                   # items.ItemCategory value
    "frame": 0                               # drop-sprite frame (mesos spin through 4)
  }]
}

ENTITY (shared by every entry; one entry per rendered instance, duplicates included):
{
  "instance_id": 3,                          # unique in the sample; entities are drawn in this order
  "bbox": [x_min, y_min, x_max, y_max] | null,
  "full_bbox": [x_min, y_min, x_max, y_max] | null,
  "visible_fraction": 0.87,
  "truncated": false,
  "flipped": true,                           # mirrored left-right
  "foot": [x, y],                            # anchor (feet / drop point) in image pixels
  "layer": 2,                                # map layer the entity was drawn on (0-7)
  "foothold_id": 41                          # platform segment it stands on (or hovers over)
}
"""

from __future__ import annotations

import bisect
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import navigation
from render.map_renderer import Bounds, DrawContext, MapRenderer, map_bounds
from scene.config import SceneConfig, parse_config
from scene.deck import Deck
from scene.sprites import MESO_IDS, EntitySprite, ItemCard, MobCard, SpriteSource, mob_state
from wz.node import WzNode
from wz.structure import WzStructure

logger = logging.getLogger(__name__)

JPEG_QUALITY = 95
FLY_HEIGHT = (40, 160)
"""Range (px) that mobs in the `fly` state hover above their platform."""

_KIND_ORDER = {"mob": 0, "character": 1, "item": 2}
_BOX_COLORS = {"character": (40, 220, 60), "mob": (235, 40, 40), "item": (250, 220, 30), "meso": (255, 140, 0)}
_MAX_REDRAWS = 20


@dataclass(frozen=True)
class Floor:
    """A walkable foothold segment (world coordinates)."""

    id: int
    layer: int
    x1: int
    y1: int
    x2: int
    y2: int

    def y_at(self, x: float) -> float:
        return self.y1 + (self.y2 - self.y1) * (x - self.x1) / (self.x2 - self.x1)


_Span = tuple[Floor, float, float]
"""A floor and the x-range of it that lies inside the viewport."""


@dataclass
class _MapInfo:
    bounds: Bounds
    floors: list[Floor]
    cumulative: list[float]


@dataclass
class _Entity:
    kind: str
    sprite: EntitySprite
    flipped: bool
    foot: tuple[int, int]
    floor: Floor
    labels: dict[str, Any]
    instance_id: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def image(self) -> Image.Image:
        return self.sprite.image

    @property
    def top_left(self) -> tuple[int, int]:
        return self.foot[0] - self.sprite.origin[0], self.foot[1] - self.sprite.origin[1]


def flip_sprite(sprite: EntitySprite) -> EntitySprite:
    """Mirror `sprite` left-right about its origin."""
    image = sprite.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return EntitySprite(image, (sprite.image.width - sprite.origin[0], sprite.origin[1]))


def box_stats(alpha: np.ndarray, top_left: tuple[int, int], owner: np.ndarray, owner_id: int) -> dict[str, Any]:
    """Boxes and visibility of one entity (see the module docstring for the definitions).

    Parameters
    ----------
    alpha
        The sprite's alpha channel, `(h, w)` uint8.
    top_left
        Sprite top-left in image pixels (may be negative or past the image).
    owner
        Final owner map, `(H, W)`; pixels equal to `owner_id` are the entity's visible pixels.
    """
    height, width = owner.shape
    opaque = alpha >= 1
    total = int(opaque.sum())
    left, top = top_left
    full_bbox = None
    truncated = False
    if total:
        ys, xs = np.nonzero(opaque)
        x0, y0, x1, y1 = left + int(xs.min()), top + int(ys.min()), left + int(xs.max()) + 1, top + int(ys.max()) + 1
        truncated = x0 < 0 or y0 < 0 or x1 > width or y1 > height
        clipped = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
        if clipped[0] < clipped[2] and clipped[1] < clipped[3]:
            full_bbox = list(clipped)
    bbox = None
    visible = 0
    if full_bbox is not None:
        cx0, cy0, cx1, cy1 = full_bbox
        region = owner[cy0:cy1, cx0:cx1] == owner_id
        visible = int(region.sum())
        if visible:
            ys, xs = np.nonzero(region)
            bbox = [cx0 + int(xs.min()), cy0 + int(ys.min()), cx0 + int(xs.max()) + 1, cy0 + int(ys.max()) + 1]
    return {
        "bbox": bbox,
        "full_bbox": full_bbox,
        "visible_fraction": round(visible / total, 4) if total else 0.0,
        "truncated": truncated,
    }


class SceneGenerator:
    """Builds scenes from the sub-APIs of a `WzApi` (see the module docstring for the output)."""

    def __init__(self, wz: WzStructure, map_api: Any, mob_api: Any, item_api: Any, character_api: Any) -> None:
        self._wz = wz
        self._maps = map_api
        self._mobs = mob_api
        self._items = item_api
        self._characters = character_api
        self.renderer = MapRenderer(wz)
        self._sprites = SpriteSource(wz, item_api.index)
        self._map_info: dict[int, _MapInfo | None] = {}
        self._mob_names: dict[int, str | None] | None = None
        self._cards: dict[str, list[Any]] = {}

    # Public -----------------------------------------------------------------------------

    def generate(self, config: str | dict[str, Any] | SceneConfig, output_root: Path) -> Path:
        """Render `config.num_samples` scenes into a new run folder and return it.

        Raises
        ------
        ValueError
            If the config is invalid or no usable map is available.
        """
        cfg = parse_config(config)
        if cfg.seed is None:
            cfg = cfg.with_seed(random.SystemRandom().randrange(2**32))
        rng = random.Random(cfg.seed)
        run_dir = self._run_dir(Path(output_root))
        (run_dir / "config.json").write_text(json.dumps(cfg.to_json(), indent=2) + "\n", encoding="utf-8")

        map_ids = list(cfg.maps) if cfg.maps is not None else self._maps.ids()
        if not map_ids:
            raise ValueError("No maps to draw scenes on")
        decks: dict[str, Deck[Any]] = {"map": Deck(map_ids, rng)}
        if cfg.mobs.max_count:
            decks["mob"] = Deck(self._card_list("mob", self._sprites.mob_cards), rng)
        if cfg.items.max_count:
            decks["meso"] = Deck(self._card_list("meso", self._sprites.meso_cards), rng)
            decks["drop"] = Deck(self._card_list("drop", self._sprites.drop_cards), rng)

        previous_map = None
        for index in range(cfg.num_samples):
            sample_id = f"{index:06d}_{rng.getrandbits(32):08x}"
            map_id = self._draw_map(decks["map"])
            if previous_map is not None and map_id != previous_map:
                self.renderer.clear_cache()
            previous_map = map_id
            self._write_sample(run_dir, sample_id, map_id, cfg, decks, rng)
            logger.info("Scene %s on map %s", sample_id, map_id)
        return run_dir

    # Setup ------------------------------------------------------------------------------

    def _draw_map(self, deck: Deck[int]) -> int:
        """Next map card that has walkable footholds (unusable maps are skipped when drawn)."""
        for _ in range(2 * len(deck)):
            map_id = deck.draw()
            if self._info(map_id) is not None:
                return map_id
        raise ValueError("No usable maps (with walkable footholds) to draw scenes on")

    def _card_list(self, key: str, build: Any) -> list[Any]:
        if key not in self._cards:
            self._cards[key] = build()
        return self._cards[key]

    @staticmethod
    def _run_dir(output_root: Path) -> Path:
        base = output_root / "Scenes" / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir, n = base, 1
        while run_dir.exists():
            run_dir = base.with_name(f"{base.name}_{n}")
            n += 1
        run_dir.mkdir(parents=True)
        return run_dir

    def _map_content(self, map_id: int) -> tuple[WzNode | None, WzNode | None]:
        node = self._maps.node(map_id)
        if node is None:
            return None, None
        link = node.get_int("info/link", 0)
        return node, (self._maps.node(link) if link else None) or node

    def _info(self, map_id: int) -> _MapInfo | None:
        if map_id not in self._map_info:
            info = None
            node, content = self._map_content(map_id)
            if content is not None:
                try:
                    floors = [
                        Floor(f["id"], f["layer"], f["x1"], f["y1"], f["x2"], f["y2"])
                        for f in navigation.footholds(content)
                        if f["x1"] != f["x2"]
                    ]
                    if floors:
                        cumulative: list[float] = []
                        total = 0.0
                        for f in floors:
                            total += abs(f.x2 - f.x1)
                            cumulative.append(total)
                        info = _MapInfo(map_bounds(content), floors, cumulative)
                except ValueError as exc:
                    logger.warning("Skipping map %s: %s", map_id, exc)
            else:
                logger.warning("Map %s not found", map_id)
            self._map_info[map_id] = info
        return self._map_info[map_id]

    def _mob_name(self, mob_id: int) -> str | None:
        if self._mob_names is None:
            self._mob_names = dict(self._mobs.list())
        return self._mob_names.get(mob_id)

    # Layout -----------------------------------------------------------------------------

    @staticmethod
    def _viewport(info: _MapInfo, size: tuple[int, int], rng: random.Random) -> Bounds:
        width, height = size
        floor = info.floors[bisect.bisect_left(info.cumulative, rng.uniform(0, info.cumulative[-1]))]
        px = rng.uniform(min(floor.x1, floor.x2), max(floor.x1, floor.x2))
        py = floor.y_at(px)
        cx = px + rng.uniform(-0.35, 0.35) * width
        cy = py - rng.uniform(-0.25, 0.25) * height - 0.1 * height
        b = info.bounds

        def axis(center: float, span: int, lo: int, hi: int) -> int:
            if hi - lo <= span:
                return lo - (span - (hi - lo)) // 2
            return int(min(max(center - span / 2, lo), hi - span))

        left = axis(cx, width, b.left, b.right)
        top = axis(cy, height, b.top, b.bottom)
        return Bounds(left, top, left + width, top + height)

    @staticmethod
    def _visible_floors(info: _MapInfo, vp: Bounds, margin: int) -> tuple[list[_Span], list[float]]:
        spans: list[_Span] = []
        cumulative: list[float] = []
        total = 0.0
        for f in info.floors:
            lo = max(min(f.x1, f.x2), vp.left + margin)
            hi = min(max(f.x1, f.x2), vp.right - margin)
            if hi <= lo:
                continue
            ya, yb = f.y_at(lo), f.y_at(hi)
            if max(ya, yb) < vp.top + margin or min(ya, yb) > vp.bottom - 2:
                continue
            spans.append((f, lo, hi))
            total += hi - lo
            cumulative.append(total)
        return spans, cumulative

    def _foot(
        self, info: _MapInfo, vp: Bounds, spans: list[_Span], cumulative: list[float], rng: random.Random
    ) -> tuple[Floor, tuple[int, int]]:
        if spans:
            for _ in range(_MAX_REDRAWS):
                floor, lo, hi = spans[bisect.bisect_left(cumulative, rng.uniform(0, cumulative[-1]))]
                x = rng.uniform(lo, hi)
                y = floor.y_at(x)
                if vp.top + 20 <= y <= vp.bottom - 2:
                    return floor, (round(x), round(y))
        floor = info.floors[bisect.bisect_left(info.cumulative, rng.uniform(0, info.cumulative[-1]))]
        x = min(max(rng.uniform(vp.left, vp.right), min(floor.x1, floor.x2)), max(floor.x1, floor.x2))
        return floor, (round(x), round(floor.y_at(x)))

    def _entities(
        self, cfg: SceneConfig, decks: dict[str, Deck[Any]], info: _MapInfo, vp: Bounds, rng: random.Random
    ) -> list[_Entity]:
        margin = 20
        spans, cumulative = self._visible_floors(info, vp, margin)
        out: list[_Entity] = []

        def add(kind: str, sprite: EntitySprite, labels: dict[str, Any], lift: int = 0) -> None:
            flipped = rng.random() < cfg.flip_probability
            if flipped:
                sprite = flip_sprite(sprite)
            floor, (x, y) = self._foot(info, vp, spans, cumulative, rng)
            out.append(_Entity(kind, sprite, flipped, (x, y - lift), floor, labels))

        for _ in range(cfg.characters.sample(rng)):
            spec = self._characters.random(rng, cfg.equip_weights)
            result = self._characters.render(spec)
            labels = {"action": result.action, "spec": spec.to_dict(), "code": spec.encode()}
            add("character", EntitySprite(result.image, result.origin), labels)

        for _ in range(cfg.mobs.sample(rng)):
            for _ in range(_MAX_REDRAWS):
                card: MobCard = decks["mob"].draw()
                sprite = self._sprites.mob_sprite(card)
                if sprite is not None:
                    break
            else:
                continue
            state = mob_state(card.action)
            labels = {
                "mob_id": card.mob_id,
                "mob_name": self._mob_name(card.mob_id),
                "source_mob_id": self._sprites.mob_source_id(card.mob_id),
                "action": card.action,
                "state": state,
                "frame": card.frame,
            }
            add("mob", sprite, labels, rng.randint(*FLY_HEIGHT) if state == "fly" else 0)

        for _ in range(cfg.items.sample(rng)):
            is_meso = rng.random() < cfg.meso_fraction
            deck = decks["meso" if is_meso else "drop"]
            for _ in range(_MAX_REDRAWS):
                item_card: ItemCard = deck.draw()
                sprite = self._sprites.item_sprite(item_card)
                if sprite is not None:
                    break
            else:
                continue
            category = self._items.category(item_card.item_id)
            labels = {
                "item_id": item_card.item_id,
                "item_name": self._items.name(item_card.item_id),
                "is_meso": item_card.item_id in MESO_IDS,
                "category": category.value if category is not None else None,
                "frame": item_card.frame,
            }
            add("item", sprite, labels)

        order = sorted(range(len(out)), key=lambda i: (out[i].floor.layer, _KIND_ORDER[out[i].kind], i))
        ordered = [out[i] for i in order]
        for instance_id, entity in enumerate(ordered, start=1):
            entity.instance_id = instance_id
        return ordered

    # Rendering --------------------------------------------------------------------------

    def _write_sample(
        self,
        run_dir: Path,
        sample_id: str,
        map_id: int,
        cfg: SceneConfig,
        decks: dict[str, Deck[Any]],
        rng: random.Random,
    ) -> None:
        info = self._info(map_id)
        node, content = self._map_content(map_id)
        vp = self._viewport(info, cfg.resolution, rng)
        entities = self._entities(cfg, decks, info, vp, rng)
        by_layer: dict[int, list[_Entity]] = {}
        for entity in entities:
            by_layer.setdefault(min(max(entity.floor.layer, 0), 7), []).append(entity)

        def hook(layer: int, ctx: DrawContext) -> None:
            for entity in by_layer.get(layer, ()):
                self.renderer.draw_entity(ctx, entity.image, *entity.top_left, entity.instance_id)

        result = self.renderer.render(node, vp, layer_hook=hook, track_owner=True)
        owner = np.asarray(result.owner)
        for entity in entities:
            left, top = entity.top_left
            alpha = np.asarray(entity.image.getchannel("A"))
            entity.stats = box_stats(alpha, (left - vp.left, top - vp.top), owner, entity.instance_id)

        image = result.image.convert("RGB")
        image.save(run_dir / f"{sample_id}.jpg", format="JPEG", quality=JPEG_QUALITY)
        self._draw_boxes(image, entities).save(run_dir / f"{sample_id}.bbox.jpg", format="JPEG", quality=JPEG_QUALITY)

        street, map_name = self._maps.names().get(map_id, (None, None))
        metadata = {
            "sample_id": sample_id,
            "image": f"{sample_id}.jpg",
            "bbox_image": f"{sample_id}.bbox.jpg",
            "resolution": list(cfg.resolution),
            "seed": cfg.seed,
            "map": {
                "id": map_id,
                "name": self._maps.name(map_id),
                "street_name": street,
                "map_name": map_name,
                "viewport": [vp.left, vp.top, vp.right, vp.bottom],
            },
            "characters": [self._entry(e, vp) for e in entities if e.kind == "character"],
            "mobs": [self._entry(e, vp) for e in entities if e.kind == "mob"],
            "items": [self._entry(e, vp) for e in entities if e.kind == "item"],
        }
        (run_dir / f"{sample_id}.metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if content is not None and content is not node:
            content.unload()
        if node is not None:
            node.unload()

    @staticmethod
    def _entry(entity: _Entity, vp: Bounds) -> dict[str, Any]:
        return {
            "instance_id": entity.instance_id,
            **entity.stats,
            "flipped": entity.flipped,
            "foot": [entity.foot[0] - vp.left, entity.foot[1] - vp.top],
            "layer": entity.floor.layer,
            "foothold_id": entity.floor.id,
            **entity.labels,
        }

    @staticmethod
    def _draw_boxes(image: Image.Image, entities: list[_Entity]) -> Image.Image:
        out = image.copy()
        draw = ImageDraw.Draw(out)
        for entity in entities:
            box = entity.stats.get("bbox")
            if box is None:
                continue
            kind = "meso" if entity.labels.get("is_meso") else entity.kind
            color = _BOX_COLORS[kind]
            draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=color, width=2)
            if entity.kind == "mob":
                label = f"{entity.labels['mob_name'] or entity.labels['mob_id']} [{entity.labels['state']}]"
            elif entity.labels.get("is_meso"):
                label = "meso"
            elif entity.kind == "item":
                label = str(entity.labels["item_name"] or entity.labels["item_id"])
            else:
                label = "character"
            text_y = max(0, box[1] - 11)
            draw.rectangle((box[0], text_y, box[0] + 6 * len(label) + 2, text_y + 10), fill=color)
            draw.text((box[0] + 1, text_y - 1), label, fill=(0, 0, 0))
        return out
