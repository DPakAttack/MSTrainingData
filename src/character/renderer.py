"""Composite a standing character from its parts (static port of `AvatarCanvas`).

Each sprite layer of a frame has an `origin`, a `z` layer name, and `map` anchor vectors
(`navel`, `neck`, `brow`, `hand`, ...) relative to its origin. Layers are chained together by
shared anchors, starting from the body whose origin is the character's foot point. Visibility
follows `Base/smap.img`: a layer is drawn only if its part owns one of the layer's slot codes
after every part has claimed its `vslot` codes in `VSLOT_PRECEDENCE` order (so a cap hides the
hair layers it covers). Layers are drawn back-to-front following `Base/zmap.img`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from PIL import Image

from character.catalog import CharacterCatalog, ItemInfo, split_slot_codes
from character.codec import CharacterSpec
from character.equip import VSLOT_PRECEDENCE, EquipType
from wz.node import WzNode
from wz.structure import WzStructure
from wz.values import WzVector

_SKIPPED_LAYERS = frozenset({"ear", "lefEar", "highlefEar", "hairShade"})
_DRAW_ORDER = (
    EquipType.SKIN,
    EquipType.FACE,
    EquipType.HAIR,
    EquipType.CAP,
    EquipType.FACE_ACCESSORY,
    EquipType.EYE_ACCESSORY,
    EquipType.EARRINGS,
    EquipType.COAT,
    EquipType.LONGCOAT,
    EquipType.PANTS,
    EquipType.SHOES,
    EquipType.GLOVE,
    EquipType.CAPE,
    EquipType.SHIELD,
    EquipType.WEAPON,
)
"""Order in which parts are attached; anchors defined earlier win."""


@dataclass
class CharacterImage:
    """A rendered character."""

    image: Image.Image
    """RGBA sprite cropped to its content."""

    origin: tuple[int, int]
    """Pixel position of the character's foot point (where it stands on a foothold)."""

    action: str
    """Body pose that was drawn (`stand1` or `stand2`)."""

    skipped: list[str] = field(default_factory=list)
    """Parts that had no frame for the pose."""


@dataclass
class _Layer:
    slot: EquipType
    name: str
    image: Image.Image
    origin: tuple[int, int]
    anchors: dict[str, tuple[int, int]]
    z: str | int
    order: int


def _vec(node: WzNode | None, name: str) -> tuple[int, int] | None:
    value = node.get_value(name) if node is not None else None
    return (value.x, value.y) if isinstance(value, WzVector) else None


def solve_positions(layers: list[_Layer]) -> list[tuple[int, int] | None]:
    """World position of each layer's origin, chaining layers through shared anchor names.

    The first layer's origin is placed at `(0, 0)`. A layer is placed by the first of its anchors
    that is already known (`origin = anchor - map[anchor]`), then contributes the anchors it
    defines that are still unknown. Layers that never connect are placed at `(0, 0)`.
    """
    known: dict[str, tuple[int, int]] = {}
    positions: list[tuple[int, int] | None] = [None] * len(layers)

    def place(i: int, origin: tuple[int, int]) -> None:
        positions[i] = origin
        for name, (mx, my) in layers[i].anchors.items():
            known.setdefault(name, (origin[0] + mx, origin[1] + my))

    progress = True
    while progress:
        progress = False
        for i, layer in enumerate(layers):
            if positions[i] is not None:
                continue
            if not known and not any(p is not None for p in positions):
                place(i, (0, 0))
                progress = True
                continue
            for name, (mx, my) in layer.anchors.items():
                if name in known:
                    kx, ky = known[name]
                    place(i, (kx - mx, ky - my))
                    progress = True
                    break
    for i, pos in enumerate(positions):
        if pos is None:
            place(i, (0, 0))
    return positions


class CharacterRenderer:
    """Renders `CharacterSpec`s.

    Parameters
    ----------
    wz
        Mounted data tree.
    catalog
        Item index (shared with the API so metadata is computed once).
    cache_size
        Number of decoded sprite bitmaps kept in memory.
    """

    def __init__(self, wz: WzStructure, catalog: CharacterCatalog, cache_size: int = 4096) -> None:
        self.wz = wz
        self.catalog = catalog
        self.cache_size = cache_size
        self._bitmaps: OrderedDict[str, Image.Image] = OrderedDict()
        self._zmap: dict[str, int] | None = None
        self._smap: dict[str, tuple[str, ...]] | None = None

    # Lookup tables -----------------------------------------------------------------------

    @property
    def zmap(self) -> dict[str, int]:
        """Layer name -> index in `Base/zmap.img` (0 is frontmost)."""
        if self._zmap is None:
            node = self.wz.find("Base/zmap.img")
            self._zmap = {c.name: i for i, c in enumerate(node or ())}
        return self._zmap

    @property
    def smap(self) -> dict[str, tuple[str, ...]]:
        """Layer name -> slot codes from `Base/smap.img`."""
        if self._smap is None:
            node = self.wz.find("Base/smap.img")
            self._smap = {c.name: split_slot_codes(c.value if isinstance(c.value, str) else None) for c in node or ()}
        return self._smap

    def _z_index(self, z: str | int) -> int:
        count = len(self.zmap)
        if isinstance(z, str):
            return self.zmap.get(z, count)
        return count - z if z < 0 else -1 - z

    def _bitmap(self, node: WzNode) -> Image.Image | None:
        target = node.resolve_canvas()
        key = target.path
        image = self._bitmaps.get(key)
        if image is None:
            canvas = target.canvas
            if canvas is None:
                return None
            image = canvas.to_image()
            self._bitmaps[key] = image
            if len(self._bitmaps) > self.cache_size:
                self._bitmaps.popitem(last=False)
        else:
            self._bitmaps.move_to_end(key)
        return image

    # Frames ------------------------------------------------------------------------------

    def _frame(self, img: WzNode, candidates: list[str]) -> WzNode | None:
        for path in candidates:
            node = img.find(path)
            if node is None:
                continue
            node = node.resolve()
            if any(c.resolve().canvas is not None for c in node):
                return node
        return None

    def _frame_candidates(self, slot: EquipType, info: ItemInfo, action: str) -> list[str]:
        prefix = f"{info.weapon_type}/" if info.weapon_type else ""
        if slot == EquipType.FACE:
            return ["default", "default/0"]
        base = [f"{prefix}{action}/0", f"{prefix}stand1/0", f"{prefix}stand2/0"]
        if slot == EquipType.HAIR:
            return [*base, "default"]
        return [*base, "default", "default/0"]

    def _layers(self, slot: EquipType, frame: WzNode, start: int) -> tuple[list[_Layer], bool]:
        layers: list[_Layer] = []
        face_visible = True
        multi_weapon = frame.get("weapon1") is not None
        for child in frame:
            if child.name in _SKIPPED_LAYERS:
                continue
            target = child.resolve()
            if target.canvas is None:
                if child.name == "face" and isinstance(target.value, int):
                    face_visible = target.value != 0
                continue
            if multi_weapon and child.name.startswith("weapon") and child.name != "weapon":
                continue
            image = self._bitmap(target)
            if image is None:
                continue
            canvas_node = target.resolve_canvas()
            origin = _vec(target, "origin") or _vec(canvas_node, "origin") or (0, 0)
            map_node = target.get("map") or canvas_node.get("map")
            anchors = {m.name: (m.value.x, m.value.y) for m in map_node or () if isinstance(m.value, WzVector)}
            anchors.pop("muzzle", None)
            z_raw = target.get_value("z", canvas_node.get_value("z", ""))
            z = z_raw if isinstance(z_raw, int) else str(z_raw)
            layers.append(_Layer(slot, child.name, image, origin, anchors, z, start + len(layers)))
        return layers, face_visible

    # Rendering ---------------------------------------------------------------------------

    def render(self, spec: CharacterSpec) -> CharacterImage:
        """Draw `spec` in its standing pose.

        Raises
        ------
        ValueError
            If a required slot is missing or an id is not a drawable item.
        """
        if spec.missing_required():
            raise ValueError(f"Missing required slots: {[s.value for s in spec.missing_required()]}")
        infos: dict[EquipType, ItemInfo] = {}
        for slot, item_id in spec.parts:
            info = self.catalog.get(slot, item_id)
            if info is None:
                raise ValueError(f"{item_id} is not a drawable {slot.value}")
            infos[slot] = info
        weapon = infos.get(EquipType.WEAPON)
        action = "stand2" if weapon is not None and weapon.two_handed else "stand1"

        owner: dict[str, EquipType] = {}
        for slot in VSLOT_PRECEDENCE:
            if slot == EquipType.SKIN:
                continue
            for code in infos[slot].vslot if slot in infos else ():
                owner[code] = slot

        skipped: list[str] = []
        layers: list[_Layer] = []
        face_visible = True
        touched: list[WzNode] = []
        skin = spec[EquipType.SKIN]
        sources: list[tuple[EquipType, WzNode | None, list[str]]] = [
            (EquipType.SKIN, self.wz.find(f"Character/{2000 + skin:08d}.img"), [f"{action}/0", "stand1/0"]),
            (EquipType.SKIN, self.wz.find(f"Character/{12000 + skin:08d}.img"), [f"{action}/0", "front"]),
        ]
        for slot in _DRAW_ORDER[1:]:
            if slot in infos:
                img = self.wz.find(infos[slot].path)
                sources.append((slot, img, self._frame_candidates(slot, infos[slot], action)))
                if img is not None:
                    touched.append(img)

        for slot, img, candidates in sources:
            frame = self._frame(img, candidates) if img is not None else None
            if frame is None:
                skipped.append(f"{slot.value} {img.path if img is not None else '?'}: no frame for {action}")
                continue
            part_layers, part_face = self._layers(slot, frame, len(layers))
            if slot == EquipType.SKIN:
                face_visible = face_visible and part_face
            layers += part_layers

        if not face_visible or any(i.invisible_face for i in infos.values() if i.slot == EquipType.FACE_ACCESSORY):
            layers = [layer for layer in layers if layer.slot != EquipType.FACE]
        layers = [layer for layer in layers if self._visible(layer, owner)]
        image, origin = self._composite(layers)
        for img in touched:
            img.unload()
        return CharacterImage(image, origin, action, skipped)

    def _visible(self, layer: _Layer, owner: dict[str, EquipType]) -> bool:
        if layer.slot == EquipType.SKIN or not isinstance(layer.z, str):
            return True
        codes = self.smap.get(layer.z)
        if not codes:
            return True
        return any(owner.get(code) == layer.slot for code in codes)

    def _composite(self, layers: list[_Layer]) -> tuple[Image.Image, tuple[int, int]]:
        positions = solve_positions(layers)
        placed = []
        for layer, pos in zip(layers, positions, strict=True):
            x = pos[0] - layer.origin[0]
            y = pos[1] - layer.origin[1]
            placed.append((self._z_index(layer.z), layer.order, x, y, layer.image))
        if not placed:
            return Image.new("RGBA", (1, 1)), (0, 0)
        left = min(p[2] for p in placed)
        top = min(p[3] for p in placed)
        right = max(p[2] + p[4].width for p in placed)
        bottom = max(p[3] + p[4].height for p in placed)
        canvas = Image.new("RGBA", (right - left, bottom - top))
        for _, _, x, y, image in sorted(placed, key=lambda p: (-p[0], p[1])):
            canvas.alpha_composite(image, dest=(x - left, y - top))
        return canvas, (-left, -top)
