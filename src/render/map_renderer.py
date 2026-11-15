"""Static map renderer: composites backgrounds, tiles, and objects of a map into one image.

This ports the subset of `WzComparerR2.MapRender` needed for a deterministic snapshot:

- Draw order: back layers (`front=0`) by index, then layers 0-7 (objects sorted by `(z, index)`,
  then tiles sorted by `(frame z, index)`), then portals, then front layers (`front=1`).
- Backgrounds use the reference parallax formula with the camera at the center of the rendered
  rectangle (the map bounds, or a viewport), and are tiled across it for tile types 1-7.
- A per-layer hook lets callers insert sprites (characters, mobs, drops) at their in-game depth.
- Portals use the in-game sprite from `Map/MapHelper.img/portal/game`; portal types without one
  (spawn points, invisible and collision portals) are not drawn, matching the client.
- Only frame 0 of animations is drawn. Life (mobs, NPCs), reactors, particles, spine backgrounds,
  light objects, and quest-gated items are skipped.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

from PIL import Image

from wz.node import WzNode
from wz.structure import WzStructure
from wz.values import WzVector

logger = logging.getLogger(__name__)

_H, _V, _SH, _SV = 1, 2, 4, 8
_TILE_MODES = {0: 0, 1: _H, 2: _V, 3: _H | _V, 4: _H | _SH, 5: _V | _SV, 6: _H | _V | _SH, 7: _H | _V | _SV}

PORTAL_TYPES = (
    "sp", "pi", "pv", "pc", "pg", "tp", "ps", "pgi", "psi", "pcs",
    "ph", "psh", "pcj", "pci", "pci2", "pcig", "pshg", "pcc", "pcir",
)  # fmt: skip
"""Portal type names indexed by the `pt` value."""

HIDDEN_PORTAL_TYPES = frozenset({10, 11, 16})
"""`ph`, `psh`, `pshg`: drawn by the client only while the player stands near them."""

OCCLUDER_ALPHA = 128
"""Minimum alpha at which a scenery pixel hides whatever was drawn beneath it (owner tracking)."""


@dataclass(frozen=True)
class Bounds:
    """World-space rectangle of a map (right/bottom exclusive)."""

    left: int
    """Leftmost world x."""

    top: int
    """Topmost world y."""

    right: int
    """Rightmost world x."""

    bottom: int
    """Bottommost world y."""

    @property
    def width(self) -> int:
        """Width in pixels."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Height in pixels."""
        return self.bottom - self.top


@dataclass
class Sprite:
    """A decoded bitmap with its anchor."""

    image: Image.Image
    """RGBA bitmap."""

    origin: tuple[int, int]
    """Anchor point inside the bitmap; the sprite is drawn at `position - origin`."""

    z: int
    """Frame-level z order."""


@dataclass
class RenderResult:
    """Output of `MapRenderer.render`."""

    image: Image.Image
    """The composited map."""

    bounds: Bounds
    """World-space rectangle covered by `image` (before scaling)."""

    scale: float
    """Scale applied to fit `max_pixels` (1.0 when not downscaled)."""

    skipped: list[str] = field(default_factory=list)
    """Human-readable notes for items that could not be drawn."""

    owner: Image.Image | None = None
    """Per-pixel id of the entity drawn last (`0` for scenery) when `track_owner` was set."""


def map_bounds(map_node: WzNode) -> Bounds:
    """Compute the world bounds of a map.

    Uses `info/VR*` when present, otherwise the extent of footholds (padded like the reference) and
    ladders/ropes, otherwise the minimap rectangle.
    """
    info = map_node.get("info")
    if info is not None and all(info.get(k) is not None for k in ("VRLeft", "VRTop", "VRRight", "VRBottom")):
        b = Bounds(info.get_int("VRLeft"), info.get_int("VRTop"), info.get_int("VRRight"), info.get_int("VRBottom"))
        if b.width > 0 and b.height > 0:
            return b

    xs: list[int] = []
    ys: list[int] = []
    fh_root = map_node.get("foothold")
    if fh_root is not None:
        for layer in fh_root:
            for group in layer:
                for fh in group:
                    xs += [fh.get_int("x1"), fh.get_int("x2")]
                    ys += [fh.get_int("y1"), fh.get_int("y2")]
    if xs:
        left, right, top, bottom = min(xs), max(xs), min(ys) - 250, max(ys) + 200
        lr = map_node.get("ladderRope")
        if lr is not None:
            for item in lr:
                x = item.get_int("x")
                left, right = min(left, x), max(right, x + 1)
                top = min(top, item.get_int("y1"), item.get_int("y2"))
                bottom = max(bottom, item.get_int("y1"), item.get_int("y2"))
        if right > left and bottom > top:
            return Bounds(left, top, right, bottom)

    mini = map_node.get("miniMap")
    if mini is not None and mini.get_int("width") > 0:
        cx, cy = mini.get_int("centerX"), mini.get_int("centerY")
        return Bounds(-cx, -cy, mini.get_int("width") - cx, mini.get_int("height") - cy)
    raise ValueError(f"Cannot determine bounds of {map_node.path}")


def _sort_index(node: WzNode) -> int:
    try:
        return int(node.name)
    except ValueError:
        return 0


def _is_quest_gated(node: WzNode) -> bool:
    return any(node.get(k) is not None for k in ("quest", "questex")) or bool(node.get_str("tags") or "")


class MapRenderer:
    """Renders maps from a mounted `WzStructure`.

    Parameters
    ----------
    wz
        Mounted data tree.
    max_pixels
        Maps larger than this many pixels are rendered downscaled to fit.
    include_quest_items
        Also draw objects/backgrounds that are only visible in some quest states.
    include_portals
        Draw portals that have an in-game sprite.
    include_hidden_portals
        Also draw hidden portals (`HIDDEN_PORTAL_TYPES`), which the client only shows up close.
    background
        RGBA fill used before compositing.
    """

    def __init__(
        self,
        wz: WzStructure,
        max_pixels: int = 64_000_000,
        include_quest_items: bool = False,
        include_portals: bool = True,
        include_hidden_portals: bool = False,
        background: tuple[int, int, int, int] = (0, 0, 0, 255),
    ) -> None:
        self.wz = wz
        self.max_pixels = max_pixels
        self.include_quest_items = include_quest_items
        self.include_portals = include_portals
        self.include_hidden_portals = include_hidden_portals
        self.background = background
        self._sprites: dict[str, Sprite | None] = {}
        self._scaled: dict[tuple[str, bool, int, float], Image.Image] = {}
        self._masks: dict[int, Image.Image] = {}

    # Public -------------------------------------------------------------------------------

    def render(
        self,
        map_node: WzNode,
        viewport: Bounds | None = None,
        layer_hook: Callable[[int, DrawContext], None] | None = None,
        track_owner: bool = False,
    ) -> RenderResult:
        """Render `map_node` (a `Map/Map/MapN/<id>.img` node) to an image.

        Parameters
        ----------
        viewport
            World rectangle to render at scale 1 (the camera is its center, so parallax matches the
            client). The whole map, fitted to `max_pixels`, when `None`.
        layer_hook
            Called as `layer_hook(layer, ctx)` after each of layers 0-7 has drawn its objects and
            tiles; use `draw_entity` there to insert sprites at that depth.
        track_owner
            Keep `RenderResult.owner`: an `I`-mode image holding, per pixel, the id of the entity
            drawn last there (`0` = scenery). Scenery pixels with alpha >= `OCCLUDER_ALPHA` and
            entity pixels with alpha >= 1 take ownership.
        """
        content = self._follow_link(map_node)
        if viewport is not None:
            bounds, scale = viewport, 1.0
        else:
            bounds = map_bounds(content)
            scale = 1.0
            if bounds.width * bounds.height > self.max_pixels:
                scale = math.sqrt(self.max_pixels / (bounds.width * bounds.height))
        size = (max(1, round(bounds.width * scale)), max(1, round(bounds.height * scale)))
        canvas = Image.new("RGBA", size, self.background)
        owner = Image.new("I", size, 0) if track_owner else None
        ctx = DrawContext(canvas, bounds, scale, [], owner)

        backs = content.get("back")
        if backs is not None:
            self._draw_backs(ctx, backs, front=False)
        for layer in range(8):
            layer_node = content.get(str(layer))
            if layer_node is not None:
                self._draw_layer(ctx, layer_node)
            if layer_hook is not None:
                layer_hook(layer, ctx)
        portals = content.get("portal")
        if portals is not None and self.include_portals:
            self._draw_portals(ctx, portals)
        if backs is not None:
            self._draw_backs(ctx, backs, front=True)
        self._scaled.clear()
        self._masks.clear()
        return RenderResult(canvas, bounds, scale, ctx.skipped, owner)

    def draw_entity(self, ctx: DrawContext, image: Image.Image, left: int, top: int, owner_id: int) -> None:
        """Composite an RGBA `image` with its top-left at world `(left, top)` (scale 1 only).

        When the context tracks ownership, pixels with alpha >= 1 are assigned `owner_id`.
        """
        dx, dy = left - ctx.bounds.left, top - ctx.bounds.top
        cw, ch = ctx.canvas.size
        sx0, sy0 = max(0, -dx), max(0, -dy)
        sx1, sy1 = min(image.width, cw - dx), min(image.height, ch - dy)
        if sx0 >= sx1 or sy0 >= sy1:
            return
        if (sx0, sy0, sx1, sy1) != (0, 0, image.width, image.height):
            image = image.crop((sx0, sy0, sx1, sy1))
        ctx.canvas.alpha_composite(image, dest=(dx + sx0, dy + sy0))
        if ctx.owner is not None:
            mask = image.getchannel("A").point(lambda v: 255 if v >= 1 else 0)
            ctx.owner.paste(owner_id, (dx + sx0, dy + sy0, dx + sx1, dy + sy1), mask)

    def clear_cache(self) -> None:
        """Forget decoded sprites."""
        self._sprites.clear()
        self._scaled.clear()

    # Helpers ------------------------------------------------------------------------------

    def _follow_link(self, map_node: WzNode) -> WzNode:
        link = map_node.get_int("info/link", 0)
        if not link:
            return map_node
        name = f"{link:09d}.img"
        target = self.wz.find(f"Map/Map/Map{name[0]}/{name}")
        return target if target is not None else map_node

    def load_sprite(self, node: WzNode | None) -> Sprite | None:
        """Decode the bitmap at `node` (following links), with origin and z. Cached by path."""
        if node is None:
            return None
        node = node.resolve()
        if node.canvas is None:
            frame = node.get("0")
            if frame is None or frame.resolve().canvas is None:
                return None
            node = frame.resolve()
        key = node.path
        if key in self._sprites:
            return self._sprites[key]
        target = node.resolve_canvas()
        canvas = target.canvas
        sprite = None
        if canvas is not None:
            origin = node.get_value("origin") or target.get_value("origin")
            ox, oy = (origin.x, origin.y) if isinstance(origin, WzVector) else (0, 0)
            z = node.get_int("z", target.get_int("z", 0))
            try:
                sprite = Sprite(canvas.to_image(), (ox, oy), z)
            except (ValueError, OSError) as exc:
                logger.warning("Failed to decode %s: %s", key, exc)
        self._sprites[key] = sprite
        return sprite

    def _draw_layer(self, ctx: DrawContext, layer: WzNode) -> None:
        objs = layer.get("obj")
        if objs is not None:
            items = []
            for obj in objs:
                if obj.get_int("light") or (not self.include_quest_items and _is_quest_gated(obj)):
                    continue
                path = "/".join(obj.get_str(k) or "" for k in ("l0", "l1", "l2"))
                sprite_node = self.wz.find(f"Map/Obj/{obj.get_str('oS')}.img/{path}")
                sprite = self.load_sprite(sprite_node)
                if sprite is None:
                    ctx.skipped.append(f"obj {obj.path}: missing sprite Map/Obj/{obj.get_str('oS')}.img/{path}")
                    continue
                items.append(((obj.get_int("z"), _sort_index(obj)), obj, sprite))
            for _, obj, sprite in sorted(items, key=lambda t: t[0]):
                self._blit(ctx, sprite, obj.get_int("x"), obj.get_int("y"), flip=bool(obj.get_int("f")))

        tiles = layer.get("tile")
        tile_set = layer.get_str("info/tS")
        if tiles is not None and tile_set:
            items = []
            for tile in tiles:
                sprite = self.load_sprite(
                    self.wz.find(f"Map/Tile/{tile_set}.img/{tile.get_str('u')}/{tile.get_int('no')}")
                )
                if sprite is None:
                    ctx.skipped.append(f"tile {tile.path}: missing sprite")
                    continue
                items.append(((sprite.z, _sort_index(tile)), tile, sprite))
            for _, tile, sprite in sorted(items, key=lambda t: t[0]):
                self._blit(ctx, sprite, tile.get_int("x"), tile.get_int("y"))

    def portal_sprite_node(self, portal: WzNode) -> WzNode | None:
        """The in-game sprite node for `portal`, or `None` if the client does not draw it."""
        pt = portal.get_int("pt")
        if pt == 7:
            pt = 2
        if not 0 <= pt < len(PORTAL_TYPES):
            return None
        image = portal.get_int("image")
        node = self.wz.find(f"Map/MapHelper.img/portal/game/{PORTAL_TYPES[pt]}/{image or 'default'}")
        if node is not None and node.get("portalContinue") is not None:
            node = node["portalContinue"]
        return node

    def _draw_portals(self, ctx: DrawContext, portals: WzNode) -> None:
        items = []
        for portal in portals:
            if portal.get_int("pt") in HIDDEN_PORTAL_TYPES and not self.include_hidden_portals:
                continue
            node = self.portal_sprite_node(portal)
            if node is None:
                continue
            sprite = self.load_sprite(node)
            if sprite is None:
                ctx.skipped.append(f"portal {portal.path}: missing sprite {node.path}")
                continue
            items.append(((sprite.z, _sort_index(portal)), portal, sprite))
        for _, portal, sprite in sorted(items, key=lambda t: t[0]):
            self._blit(ctx, sprite, portal.get_int("x"), portal.get_int("y"))

    def _draw_backs(self, ctx: DrawContext, backs: WzNode, front: bool) -> None:
        bounds = ctx.bounds
        cam_x = (bounds.left + bounds.right) / 2
        cam_y = (bounds.top + bounds.bottom) / 2
        for back in sorted(backs, key=_sort_index):
            if bool(back.get_int("front")) != front or back.get_int("screenMode") != 0:
                continue
            if not self.include_quest_items and back.get_str("backTags"):
                continue
            ani = back.get_int("ani")
            bs = back.get_str("bS")
            if not bs:
                continue
            if ani == 2 or back.get_str("spineAni"):
                ctx.skipped.append(f"back {back.path}: spine background not supported")
                continue
            kind = "back" if ani == 0 else "ani"
            sprite = self.load_sprite(self.wz.find(f"Map/Back/{bs}.img/{kind}/{back.get_str('no')}"))
            if sprite is None:
                ctx.skipped.append(f"back {back.path}: missing sprite Map/Back/{bs}.img/{kind}/{back.get_str('no')}")
                continue
            self._draw_back(ctx, back, sprite, cam_x, cam_y)

    def _draw_back(self, ctx: DrawContext, back: WzNode, sprite: Sprite, cam_x: float, cam_y: float) -> None:
        mode = _TILE_MODES.get(back.get_int("type"), 0)
        flip = bool(back.get_int("f"))
        w, h = sprite.image.size
        cx, cy = back.get_int("cx"), back.get_int("cy")
        if mode & (_H | _V):
            cx = cx or w
            cy = cy or h
        x = float(back.get_int("x"))
        y = float(back.get_int("y"))
        rx, ry = back.get_int("rx"), back.get_int("ry")
        if not mode & _SH:
            x += cam_x * (100 + rx) / 100
        if not mode & _SV:
            y += cam_y * (100 + ry) / 100
        x, y = math.floor(x), math.floor(y)

        ox, oy = sprite.origin
        bl, bt = -ox, -oy
        if flip:
            bl = -(w - ox)
        br, bb = bl + w, bt + h
        b = ctx.bounds
        if mode & _H and cx > 0:
            i0 = math.floor((b.left - x - br) / cx) - 1
            i1 = math.ceil((b.right - x - bl) / cx) + 2
        else:
            i0, i1 = 0, 1
        if mode & _V and cy > 0:
            j0 = math.floor((b.top - y - bb) / cy) - 1
            j1 = math.ceil((b.bottom - y - bt) / cy) + 2
        else:
            j0, j1 = 0, 1
        alpha = back.get_int("a", 255)
        for j in range(j0, j1):
            for i in range(i0, i1):
                self._blit(ctx, sprite, x + i * cx, y + j * cy, flip=flip, alpha=alpha)

    def _sprite_image(self, sprite: Sprite, flip: bool, alpha: int, scale: float) -> Image.Image:
        key = (str(id(sprite)), flip, alpha, scale)
        img = self._scaled.get(key)
        if img is None:
            img = sprite.image
            if flip:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if alpha < 255:
                a = img.getchannel("A").point(lambda v: v * alpha // 255)
                img = img.copy()
                img.putalpha(a)
            if scale != 1.0:
                img = img.resize(
                    (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.BILINEAR
                )
            self._scaled[key] = img
        return img

    def _blit(self, ctx: DrawContext, sprite: Sprite, x: float, y: float, flip: bool = False, alpha: int = 255) -> None:
        ox, oy = sprite.origin
        w = sprite.image.width
        left = x - (w - ox if flip else ox)
        top = y - oy
        s = ctx.scale
        img = self._sprite_image(sprite, flip, alpha, s)
        dx = round((left - ctx.bounds.left) * s)
        dy = round((top - ctx.bounds.top) * s)
        cw, ch = ctx.canvas.size
        sx0, sy0 = max(0, -dx), max(0, -dy)
        sx1, sy1 = min(img.width, cw - dx), min(img.height, ch - dy)
        if sx0 >= sx1 or sy0 >= sy1:
            return
        full = img
        if (sx0, sy0, sx1, sy1) != (0, 0, img.width, img.height):
            img = img.crop((sx0, sy0, sx1, sy1))
        ctx.canvas.alpha_composite(img, dest=(dx + sx0, dy + sy0))
        if ctx.owner is not None:
            mask = self._masks.get(id(full))
            if mask is None:
                mask = full.getchannel("A").point(lambda v: 255 if v >= OCCLUDER_ALPHA else 0)
                self._masks[id(full)] = mask
            if (sx0, sy0, sx1, sy1) != (0, 0, full.width, full.height):
                mask = mask.crop((sx0, sy0, sx1, sy1))
            ctx.owner.paste(0, (dx + sx0, dy + sy0, dx + sx1, dy + sy1), mask)


@dataclass
class DrawContext:
    """Target of one `render` call (passed to `layer_hook`)."""

    canvas: Image.Image
    bounds: Bounds
    scale: float
    skipped: list[str]
    owner: Image.Image | None = None
