"""Programmatic API for extracting mobs, maps, items, and characters from a MapleStory `Data` folder.

Example
-------
>>> from api import WzApi
>>> with WzApi("C:/Games/MapleStory/Data", "out") as wz:
...     wz.mob.list()[:2]
[(1, 'Tutorial Jr. Sentinel'), (2, 'Snail')]
...     wz.map.dump([10])
[PosixPath('out/Map/Maple Road - Mushroom Town Training Camp - 10')]
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import items
import navigation
from character.build import normalize_weights, resolve_assignments, sample_optional, sample_styled
from character.catalog import CharacterCatalog
from character.codec import CharacterSpec, decode
from character.equip import (
    GENDER_FEMALE,
    GENDER_MALE,
    GENDER_UNISEX,
    OPTIONAL_SLOTS,
    EquipType,
    equip_type_of,
    face_style,
    hair_style,
)
from character.renderer import CharacterImage, CharacterRenderer
from render.map_renderer import MapRenderer
from scene.config import SceneConfig
from scene.generator import SceneGenerator
from wz.node import WzNode
from wz.structure import WzStructure
from wz.values import WzCanvas, WzConvex, WzRawData, WzSound, WzUol, WzVector, WzVideo

logger = logging.getLogger(__name__)

_MOB_IMG_RE = re.compile(r"^(\d{7})\.img$")
_MAP_IMG_RE = re.compile(r"^(\d{9})\.img$")
_MAP_FOLDER_RE = re.compile(r"^Map\d$")
_LINK_KEYS = frozenset({"_inlink", "_outlink", "source", "_hash"})
_INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
)


def sanitize_filename(name: str, fallback: str = "_") -> str:
    """Make `name` safe to use as a single path component on both Windows and Linux.

    Parameters
    ----------
    name
        Arbitrary text, e.g. a mob name.
    fallback
        Returned when nothing usable is left.

    Returns
    -------
    `name` with characters invalid on Windows replaced by `_`, trailing dots/spaces stripped, and
    reserved device names (`CON`, `COM1`, ...) suffixed with `_`.
    """
    cleaned = _INVALID_CHARS_RE.sub("_", name).strip().rstrip(". ")
    if not cleaned:
        return fallback
    if cleaned.split(".")[0].upper() in _RESERVED_NAMES:
        cleaned += "_"
    return cleaned


def node_value_to_json(value: Any) -> Any:
    """Convert a primitive WZ value into a JSON-serializable object."""
    if isinstance(value, WzVector):
        return {"x": value.x, "y": value.y}
    if isinstance(value, WzConvex):
        return [{"x": p.x, "y": p.y} for p in value.points]
    if isinstance(value, WzUol):
        return {"_uol": value.path}
    if isinstance(value, WzCanvas):
        return {"_canvas": {"width": value.width, "height": value.height}}
    if isinstance(value, WzSound):
        return {"_sound": {"duration_ms": value.duration_ms, "length": value.data_length}}
    if isinstance(value, (WzRawData, WzVideo)):
        return {"_binary": type(value).__name__}
    if isinstance(value, bytes):
        return value.hex()
    return value


def node_to_json(node: WzNode) -> Any:
    """Serialize a node subtree (values only, no bitmaps) into nested dicts."""
    if len(node) == 0:
        return node_value_to_json(node.value)
    out: dict[str, Any] = {}
    if node.value is not None:
        out["_value"] = node_value_to_json(node.value)
    for child in node:
        if child.name not in _LINK_KEYS:
            out[child.name] = node_to_json(child)
    return out


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_png(image: Image.Image, path: Path, compress_level: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=compress_level)


class _TreeExporter:
    """Writes every bitmap under a node as PNGs mirroring the tree and returns the metadata."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.count = 0
        self._decoded: dict[str, Image.Image] = {}

    def export(self, children: Iterable[WzNode], rel: tuple[str, ...] = ()) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for child in children:
            if child.name in _LINK_KEYS:
                continue
            try:
                out[child.name] = self._export_child(child, (*rel, sanitize_filename(child.name)))
            except (ValueError, OSError, KeyError) as exc:
                logger.warning("Failed to export %s: %s", child.path, exc)
                out[child.name] = {"_error": str(exc)}
        return out

    def _export_child(self, child: WzNode, rel: tuple[str, ...]) -> Any:
        target = child.resolve()
        canvas_node = target.resolve_canvas()
        if isinstance(canvas_node.value, WzCanvas):
            rel_png = "/".join(rel) + ".png"
            key = canvas_node.path
            image = self._decoded.get(key)
            if image is None:
                image = canvas_node.value.to_image()
                self._decoded[key] = image
            _save_png(image, self.out_dir.joinpath(*rel[:-1], rel[-1] + ".png"))
            self.count += 1
            entry: dict[str, Any] = {"_image": rel_png, "_size": [image.width, image.height]}
            if isinstance(child.value, WzUol):
                entry["_uol"] = child.value.path
            entry.update(self.export(target, rel))
            return entry
        if isinstance(child.value, WzUol):
            return {"_uol": child.value.path}
        if len(target) > 0 and target is child:
            return self.export(child, rel)
        return node_value_to_json(target.value)


class MobApi:
    """Mob listing and extraction (`Mob/*.img`)."""

    def __init__(self, wz: WzStructure, output_root: Path | None) -> None:
        self._wz = wz
        self._output_root = output_root

    def ids(self) -> list[int]:
        """All mob ids present in `Mob/`, ascending."""
        mob = self._wz.find("Mob")
        if mob is None:
            return []
        return sorted(int(m.group(1)) for c in mob if (m := _MOB_IMG_RE.match(c.name)))

    def name(self, mob_id: int) -> str | None:
        """Display name from `String/Mob.img`, or `None` if it has no entry."""
        return self._wz.find("String/Mob.img").get_str(f"{mob_id}/name") if self._wz.find("String/Mob.img") else None

    def list(self) -> list[tuple[int, str | None]]:
        """Return `(mob_id, mob_name)` for every mob, sorted by id."""
        strings = self._wz.find("String/Mob.img")
        names: dict[int, str | None] = {}
        if strings is not None:
            for entry in strings:
                if entry.name.isdigit():
                    names[int(entry.name)] = entry.get_str("name")
        return [(mob_id, names.get(mob_id)) for mob_id in self.ids()]

    def node(self, mob_id: int) -> WzNode | None:
        """The `Mob/<id>.img` node."""
        return self._wz.find(f"Mob/{mob_id:07d}.img")

    def dump(self, ids: Iterable[int] | None = None, output_root: Path | None = None) -> list[Path]:
        """Export mobs to `<output_root>/Mob/<name> - <id>/`.

        Each folder mirrors the image tree: every bitmap is written as `<path>.png` (UOLs and
        `_outlink`s resolved) and `info.json` holds all values (origins, delays, info, ...).

        Parameters
        ----------
        ids
            Mob ids to export; all mobs when `None`.
        output_root
            Overrides the root passed to `WzApi`.

        Returns
        -------
        The folders written, in id order. Mobs that fail are logged and skipped.
        """
        root = _require_output(output_root or self._output_root)
        names = dict(self.list())
        written: list[Path] = []
        for mob_id in sorted(set(ids)) if ids is not None else self.ids():
            node = self.node(mob_id)
            if node is None:
                logger.warning("Mob %s not found", mob_id)
                continue
            name = names.get(mob_id)
            out_dir = root / "Mob" / sanitize_filename(f"{name or 'unknown'} - {mob_id}")
            try:
                written.append(self._dump_one(mob_id, name, node, out_dir))
            except (ValueError, OSError) as exc:
                logger.error("Failed to dump mob %s: %s", mob_id, exc)
            finally:
                node.unload()
        return written

    def _dump_one(self, mob_id: int, name: str | None, node: WzNode, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        link = node.get_int("info/link", 0)
        source = self.node(link) if link else None
        if link and source is None:
            logger.warning("Mob %s links to missing mob %s", mob_id, link)
        exporter = _TreeExporter(out_dir)
        tree: dict[str, Any] = {}
        info = node.get("info")
        if info is not None:
            tree["info"] = exporter.export(info, ("info",))
        tree.update(exporter.export(c for c in (source or node) if c.name != "info"))
        _write_json(
            out_dir / "info.json",
            {"id": mob_id, "name": name, "link": link or None, "image_count": exporter.count, "tree": tree},
        )
        if source is not None:
            source.unload()
        logger.info("Mob %s (%s): %d images -> %s", mob_id, name, exporter.count, out_dir)
        return out_dir


class MapApi:
    """Map listing, rendering, and minimap extraction (`Map/Map/MapN/*.img`)."""

    def __init__(self, wz: WzStructure, output_root: Path | None) -> None:
        self._wz = wz
        self._output_root = output_root
        self._renderer = MapRenderer(wz)
        self._names: dict[int, tuple[str | None, str | None]] | None = None
        self._portal_tables: dict[int, navigation.PortalTable | None] = {}
        self.png_compress_level = 1
        """zlib level for `map.png`; maps are large, so speed is favored over size."""

    @property
    def renderer(self) -> MapRenderer:
        """The renderer used by `dump` (configure `max_pixels` etc. here)."""
        return self._renderer

    def ids(self) -> list[int]:
        """All map ids present under `Map/Map/Map0-9`, ascending."""
        root = self._wz.find("Map/Map")
        if root is None:
            return []
        out: list[int] = []
        for folder in root:
            if _MAP_FOLDER_RE.match(folder.name):
                out += [int(m.group(1)) for c in folder if (m := _MAP_IMG_RE.match(c.name))]
        return sorted(out)

    def names(self) -> dict[int, tuple[str | None, str | None]]:
        """`map_id -> (streetName, mapName)` from every region of `String/Map.img`."""
        if self._names is None:
            self._names = {}
            strings = self._wz.find("String/Map.img")
            for region in strings or ():
                for entry in region:
                    if entry.name.isdigit():
                        self._names[int(entry.name)] = (entry.get_str("streetName"), entry.get_str("mapName"))
        return self._names

    def name(self, map_id: int) -> str | None:
        """Display name `"<streetName> - <mapName>"` (either part may be absent)."""
        street, map_name = self.names().get(map_id, (None, None))
        parts = [p for p in (street, map_name) if p]
        return " - ".join(parts) if parts else None

    def list(self) -> list[tuple[int, str | None]]:
        """Return `(map_id, map_name)` for every map, sorted by id."""
        return [(map_id, self.name(map_id)) for map_id in self.ids()]

    def node(self, map_id: int) -> WzNode | None:
        """The `Map/Map/Map<first digit>/<id>.img` node."""
        name = f"{map_id:09d}.img"
        return self._wz.find(f"Map/Map/Map{name[0]}/{name}")

    def portal_table(self, map_id: int) -> navigation.PortalTable | None:
        """`portal name -> (target_map, target_portal)` of `map_id` (following `info/link`), cached."""
        if map_id not in self._portal_tables:
            node = self.node(map_id)
            table = None
            if node is not None:
                link = node.get_int("info/link", 0)
                content = (self.node(link) if link else None) or node
                table = navigation.portal_table(content)
                if content is not node:
                    content.unload()
                node.unload()
            self._portal_tables[map_id] = table
        return self._portal_tables[map_id]

    def dump(self, ids: Iterable[int] | None = None, output_root: Path | None = None) -> list[Path]:
        """Export maps to `<output_root>/Map/<name> - <id>/`.

        Writes `map.png` (backgrounds, tiles, objects, and visible portals; no life or reactors),
        `minimap.png` (when the map has one), and `info.json` (map info, bounds, and the
        footholds, ladders, and portal links described in `navigation`).

        Parameters
        ----------
        ids
            Map ids to export; all maps when `None`.
        output_root
            Overrides the root passed to `WzApi`.

        Returns
        -------
        The folders written, in id order. Maps that fail are logged and skipped.
        """
        root = _require_output(output_root or self._output_root)
        written: list[Path] = []
        for map_id in sorted(set(ids)) if ids is not None else self.ids():
            node = self.node(map_id)
            if node is None:
                logger.warning("Map %s not found", map_id)
                continue
            name = self.name(map_id)
            out_dir = root / "Map" / sanitize_filename(f"{name or 'unknown'} - {map_id}")
            try:
                written.append(self._dump_one(map_id, node, out_dir))
            except (ValueError, OSError) as exc:
                logger.error("Failed to dump map %s: %s", map_id, exc)
            finally:
                node.unload()
        return written

    def _dump_one(self, map_id: int, node: WzNode, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        result = self._renderer.render(node)
        image = result.image
        if self._renderer.background[3] == 255:
            image = image.convert("RGB")
        _save_png(image, out_dir / "map.png", self.png_compress_level)
        for note in result.skipped:
            logger.debug("Map %s: %s", map_id, note)

        link = node.get_int("info/link", 0)
        content = self.node(link) if link else node
        content = content or node
        minimap_meta = None
        mini = content.get("miniMap")
        if mini is not None:
            minimap_meta = {k: c.value for k in ("width", "height", "centerX", "centerY", "mag") if (c := mini.get(k))}
            canvas_node = mini.get("canvas")
            canvas = canvas_node.canvas if canvas_node is not None else None
            if canvas is not None:
                _save_png(canvas.to_image(), out_dir / "minimap.png")

        street, map_name = self.names().get(map_id, (None, None))
        b = result.bounds
        _write_json(
            out_dir / "info.json",
            {
                "id": map_id,
                "street_name": street,
                "map_name": map_name,
                "link": link or None,
                "bounds": {"left": b.left, "top": b.top, "right": b.right, "bottom": b.bottom},
                "image_size": list(result.image.size),
                "image_scale": result.scale,
                "world_origin_px": [round(-b.left * result.scale), round(-b.top * result.scale)],
                "minimap": minimap_meta,
                "skipped": result.skipped,
                "info": node_to_json(node["info"]) if "info" in node else None,
                "footholds": navigation.footholds(content),
                "ladders": navigation.ladders(content),
                "portals": navigation.portals(content, map_id, self.portal_table),
            },
        )
        if content is not node:
            content.unload()
        logger.info("Map %s: %dx%d -> %s", map_id, *result.image.size, out_dir)
        return out_dir


def _require_output(root: Path | None) -> Path:
    if root is None:
        raise ValueError("output_root is required for dump operations")
    return root


Assignment = tuple["EquipType | None", int]


class CharacterApi:
    """Character sprite composition from equipment slot ids (`Character/`).

    Parameters
    ----------
    wz
        Mounted data tree.
    output_root
        Where `dump` methods write.
    catalog_cache
        Optional JSON file caching the item index (building it scans every item, ~30 s).
    default_weight
        Probability used by `random` for optional slots missing from `weight_map`.
    """

    def __init__(
        self,
        wz: WzStructure,
        output_root: Path | None,
        catalog_cache: Path | None = None,
        default_weight: float = 0.5,
    ) -> None:
        self._wz = wz
        self._output_root = output_root
        self.catalog = CharacterCatalog(wz, catalog_cache)
        self.renderer = CharacterRenderer(wz, self.catalog)
        self.default_weight = default_weight
        self._pools: dict[tuple[EquipType, int | None], list[int]] = {}

    def ids(self, slot: EquipType) -> list[int]:
        """Drawable ids for `slot`, ascending (skin tone indices 0-6 for `EquipType.SKIN`)."""
        return self.catalog.ids(slot)

    def pool(self, slot: EquipType, gender: int | None = None) -> list[int]:
        """`ids(slot)`, optionally restricted to unisex items and items of `gender`."""
        key = (slot, gender)
        if key not in self._pools:
            ids = self.ids(slot)
            if gender is not None and slot != EquipType.SKIN:
                ids = [i for i in ids if self.catalog.get(slot, i).gender in (gender, GENDER_UNISEX)]
            self._pools[key] = ids
        return self._pools[key]

    def build(self, assignments: Iterable[Assignment] | Mapping[EquipType, int]) -> CharacterSpec:
        """Combine assignments into a conflict-free spec; later assignments win conflicts.

        Parameters
        ----------
        assignments
            `(slot, id)` pairs applied in order, or a `{slot: id}` mapping (applied in `EquipType`
            order). A `None` slot is inferred from the id. Skin uses `(EquipType.SKIN, 0..6)`.

        Raises
        ------
        ValueError
            If an id is not drawable, or skin, face, or hair end up unassigned.
        """
        pairs = assignments.items() if isinstance(assignments, Mapping) else assignments
        resolved = []
        for slot, item_id in pairs:
            resolved.append((slot if slot is not None else equip_type_of(item_id), int(item_id)))
        if isinstance(assignments, Mapping):
            resolved.sort(key=lambda p: list(EquipType).index(p[0]))
        spec = resolve_assignments(resolved, self.catalog.get)
        missing = spec.missing_required()
        if missing:
            raise ValueError(f"Missing required slots: {[s.value for s in missing]}")
        return spec

    def random(
        self,
        seed: int | random.Random | None = None,
        weight_map: Mapping[EquipType | str, float] | None = None,
        consistent_gender: bool = True,
    ) -> CharacterSpec:
        """Generate a random character.

        Skin is uniform over tones; hair and face pick a style uniformly and then a color of that
        style uniformly. Each optional slot is filled with probability `weight_map[slot]`
        (`default_weight` when absent), visiting slots in a shuffled order so that conflicting
        slots (e.g. overall vs. top/bottom) are not biased; conflicts resolve as in `build`.

        Parameters
        ----------
        seed
            Seed or `random.Random` for reproducible output; `None` for fresh randomness.
        weight_map
            Per optional slot fill probability in `[0, 1]`.
        consistent_gender
            Sample a gender and only use unisex items or items of that gender.
        """
        rng = seed if isinstance(seed, random.Random) else random.Random(seed)
        weights = normalize_weights(weight_map, self.default_weight)
        gender = rng.choice((GENDER_MALE, GENDER_FEMALE)) if consistent_gender else None
        assignments: list[tuple[EquipType, int]] = [
            (EquipType.SKIN, rng.choice(self.pool(EquipType.SKIN))),
            (EquipType.FACE, sample_styled(rng, self.pool(EquipType.FACE, gender), face_style)),
            (EquipType.HAIR, sample_styled(rng, self.pool(EquipType.HAIR, gender), hair_style)),
        ]
        pools = {slot: self.pool(slot, gender) for slot in OPTIONAL_SLOTS}
        assignments += sample_optional(rng, pools, weights)
        return self.build(assignments)

    def render(self, spec: CharacterSpec | str) -> CharacterImage:
        """Render a spec (or an encoded name / filename) in its standing pose."""
        return self.renderer.render(spec if isinstance(spec, CharacterSpec) else decode(spec))

    def dump(self, specs: Iterable[CharacterSpec | str], output_root: Path | None = None) -> list[Path]:
        """Write each spec to `<output_root>/Character/<spec.encode()>.png`.

        The PNG carries text chunks `origin` (`"x,y"`, the foot point in pixels) and `spec`
        (JSON `{slot: id}`); the filename alone is enough to recover the spec via `decode`.

        Returns
        -------
        The files written, in input order. Specs that fail to render are logged and skipped.
        """
        out_dir = _require_output(output_root or self._output_root) / "Character"
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for spec in specs:
            spec = spec if isinstance(spec, CharacterSpec) else decode(spec)
            try:
                result = self.render(spec)
            except (ValueError, OSError) as exc:
                logger.error("Failed to render character %s: %s", spec.to_dict(), exc)
                continue
            meta = PngInfo()
            meta.add_text("origin", f"{result.origin[0]},{result.origin[1]}")
            meta.add_text("action", result.action)
            meta.add_text("spec", json.dumps(spec.to_dict(), sort_keys=True))
            path = out_dir / f"{spec.encode()}.png"
            result.image.save(path, format="PNG", pnginfo=meta)
            written.append(path)
        return written

    def dump_random(
        self,
        count: int,
        seed: int | None = None,
        weight_map: Mapping[EquipType | str, float] | None = None,
        consistent_gender: bool = True,
        output_root: Path | None = None,
    ) -> list[Path]:
        """Generate `count` random characters from one seeded stream and `dump` them.

        Identical characters map to the same file, so fewer than `count` files may result.
        """
        rng = random.Random(seed)
        specs = [self.random(rng, weight_map, consistent_gender) for _ in range(count)]
        return self.dump(specs, output_root)


def _has_canvas(node: WzNode) -> bool:
    return isinstance(node.resolve().value, WzCanvas) or any(_has_canvas(c) for c in node if c.name not in _LINK_KEYS)


class ItemApi:
    """Item listing and ground-drop sprite extraction.

    Covers `Item/{Consume,Install,Etc,Cash,Pet,Special}` and droppable equipment under
    `Character/` (see `items.EQUIP_FOLDERS`).
    """

    def __init__(self, wz: WzStructure, output_root: Path | None) -> None:
        self._wz = wz
        self._output_root = output_root
        self._index: dict[int, items.ItemRef] | None = None
        self._strings: dict[int, tuple[str | None, str | None]] | None = None

    @property
    def index(self) -> dict[int, items.ItemRef]:
        """Location of every item, keyed by id (built on first use)."""
        if self._index is None:
            self._index = items.index_items(self._wz)
        return self._index

    def _string(self, item_id: int) -> tuple[str | None, str | None]:
        if self._strings is None:
            self._strings = items.load_strings(self._wz)
        name, desc = self._strings.get(item_id, (None, None))
        if name is None and (ref := self.index.get(item_id)) is not None and ref.category == items.ItemCategory.SPECIAL:
            node = self._wz.find(ref.path)
            name = node.get_str("name") if node is not None else None
        return name, desc

    def ids(self, category: items.ItemCategory | Iterable[items.ItemCategory] | None = None) -> list[int]:
        """Item ids, ascending, optionally restricted to one or more categories."""
        if category is None:
            return list(self.index)
        wanted = {category} if isinstance(category, items.ItemCategory) else set(category)
        return [item_id for item_id, ref in self.index.items() if ref.category in wanted]

    def category(self, item_id: int) -> items.ItemCategory | None:
        """Category of `item_id`, or `None` if it is unknown."""
        ref = self.index.get(item_id)
        return ref.category if ref is not None else None

    def name(self, item_id: int) -> str | None:
        """Display name from `String/`, or `None` if it has no entry."""
        return self._string(item_id)[0]

    def list(
        self, category: items.ItemCategory | Iterable[items.ItemCategory] | None = None
    ) -> list[tuple[int, str | None]]:
        """Return `(item_id, item_name)` for every item (optionally per category), sorted by id."""
        return [(item_id, self.name(item_id)) for item_id in self.ids(category)]

    def node(self, item_id: int) -> WzNode | None:
        """The item node (an `.img` for equipment and pets, an entry inside one otherwise)."""
        ref = self.index.get(item_id)
        return self._wz.find(ref.path) if ref is not None else None

    def dump(
        self,
        ids: Iterable[int] | None = None,
        categories: Iterable[items.ItemCategory] | None = None,
        output_root: Path | None = None,
    ) -> list[Path]:
        """Export items to `<output_root>/Item/<Category>/<name> - <id>/`.

        Each folder holds `drop/<n>.png` (the frames drawn while the item lies on the ground,
        one for most items and several for meso), `icon.png` (the inventory icon, when present),
        and `info.json` with the frame origins and delays plus the item's non-bitmap `info` values.

        Parameters
        ----------
        ids
            Item ids to export; all items (of `categories`, if given) when `None`.
        categories
            Restrict the export to these categories.
        output_root
            Overrides the root passed to `WzApi`.

        Returns
        -------
        The folders written, in id order. Items that fail or have no sprite are logged and skipped.
        """
        root = _require_output(output_root or self._output_root)
        wanted = set(categories) if categories is not None else None
        written: list[Path] = []
        loaded: WzNode | None = None
        for item_id in sorted(set(ids)) if ids is not None else self.ids():
            ref = self.index.get(item_id)
            if ref is None:
                logger.warning("Item %s not found", item_id)
                continue
            if wanted is not None and ref.category not in wanted:
                continue
            image = self._wz.find(ref.image_path)
            if loaded is not None and image is not loaded:
                loaded.unload()
            loaded = image
            node = self._wz.find(ref.path)
            if node is None:
                logger.warning("Item %s not found at %s", item_id, ref.path)
                continue
            name, desc = self._string(item_id)
            out_dir = root / "Item" / ref.category.value / sanitize_filename(f"{name or 'unknown'} - {item_id}")
            try:
                path = self._dump_one(item_id, ref.category, name, desc, node, out_dir)
            except (ValueError, OSError) as exc:
                logger.error("Failed to dump item %s: %s", item_id, exc)
                continue
            if path is not None:
                written.append(path)
        if loaded is not None:
            loaded.unload()
        return written

    def _dump_one(
        self,
        item_id: int,
        category: items.ItemCategory,
        name: str | None,
        desc: str | None,
        node: WzNode,
        out_dir: Path,
    ) -> Path | None:
        frames = items.drop_frames(node)
        if not frames:
            logger.warning("Item %s (%s) has no drop sprite", item_id, name)
            return None
        drop: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            image = frame.resolve_canvas().value.to_image()
            rel = f"drop/{index}.png"
            _save_png(image, out_dir / "drop" / f"{index}.png")
            origin = frame.get_value("origin")
            drop.append(
                {
                    "image": rel,
                    "size": [image.width, image.height],
                    "origin": [origin.x, origin.y] if isinstance(origin, WzVector) else [0, 0],
                    "delay": frame.get_int("delay", 0) or items.DEFAULT_FRAME_DELAY,
                }
            )
        icon_rel = None
        icon = items.inventory_icon(node)
        if icon is not None:
            icon_rel = "icon.png"
            _save_png(icon.resolve_canvas().value.to_image(), out_dir / icon_rel)
        info_node = node.get("info") or node
        info = {c.name: node_to_json(c) for c in info_node if c.name not in _LINK_KEYS and not _has_canvas(c)}
        _write_json(
            out_dir / "info.json",
            {
                "id": item_id,
                "name": name,
                "desc": desc,
                "category": category.value,
                "icon": icon_rel,
                "drop": drop,
                "info": info,
            },
        )
        logger.info("Item %s (%s): %d drop frames -> %s", item_id, name, len(drop), out_dir)
        return out_dir


class WzApi:
    """Entry point: mounts a `Data` folder and exposes `mob` / `map` / `item` / `character` sub-APIs.

    Parameters
    ----------
    root_wz
        The client's `Data` directory (containing `Base`, `Map`, `Mob`, `String`, ...).
    output_root
        Where `dump` methods write; may be omitted for read-only use.
    character_catalog_cache
        Optional JSON file caching the character item index between runs.
    """

    def __init__(
        self,
        root_wz: str | Path,
        output_root: str | Path | None = None,
        character_catalog_cache: str | Path | None = None,
    ) -> None:
        self.wz = WzStructure(Path(root_wz))
        out = Path(output_root) if output_root is not None else None
        self.mob = MobApi(self.wz, out)
        self.map = MapApi(self.wz, out)
        self.item = ItemApi(self.wz, out)
        cache = Path(character_catalog_cache) if character_catalog_cache is not None else None
        self.character = CharacterApi(self.wz, out, cache)
        self._output_root = out
        self._scenes: SceneGenerator | None = None

    def generate_scene(
        self, config: str | Mapping[str, Any] | SceneConfig, output_root: str | Path | None = None
    ) -> Path:
        """Render synthetic labeled scenes into `<output_root>/Scenes/<timestamp>/`.

        `config` is a JSON string or mapping (see `scene.config`); the files written and the
        metadata format are documented in `scene.generator`.

        Returns
        -------
        The run folder.

        Raises
        ------
        ValueError
            If the config is invalid or `output_root` is missing.
        """
        root = _require_output(Path(output_root) if output_root is not None else self._output_root)
        if self._scenes is None:
            self._scenes = SceneGenerator(self.wz, self.map, self.mob, self.item, self.character)
        return self._scenes.generate(config if isinstance(config, (str, SceneConfig)) else dict(config), root)

    def node(self, path: str) -> WzNode | None:
        """Look up any node by slash-separated path, e.g. `String/Mob.img/100100/name`."""
        return self.wz.find(path)

    def close(self) -> None:
        """Release all open files."""
        self.wz.close()

    def __enter__(self) -> WzApi:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
