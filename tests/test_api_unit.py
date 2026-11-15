"""Unit tests for API helpers and renderer geometry that need no game data."""

from __future__ import annotations

from PIL import Image

from api import node_to_json, sanitize_filename
from render.map_renderer import Bounds, DrawContext, MapRenderer, Sprite, map_bounds
from wz.node import WzNode
from wz.values import WzVector


def _tree(spec: dict) -> WzNode:
    root = WzNode("")

    def fill(node: WzNode, items: dict) -> None:
        for name, value in items.items():
            child = node.add(name)
            if isinstance(value, dict):
                fill(child, value)
            else:
                child.value = value

    fill(root, spec)
    return root


def test_sanitize_filename() -> None:
    assert sanitize_filename('A<b>:c"d/e\\f|g?h*') == "A_b__c_d_e_f_g_h_"
    assert sanitize_filename("name. ") == "name"
    assert sanitize_filename("CON") == "CON_"
    assert sanitize_filename("com1.txt") == "com1.txt_"
    assert sanitize_filename("   ") == "_"
    assert sanitize_filename("Snail - 2") == "Snail - 2"


def test_node_to_json() -> None:
    root = _tree({"info": {"origin": WzVector(1, 2), "delay": 100, "_outlink": "x/y"}})
    assert node_to_json(root["info"]) == {"origin": {"x": 1, "y": 2}, "delay": 100}


def test_map_bounds_prefers_vr() -> None:
    root = _tree({"m": {"info": {"VRLeft": -10, "VRTop": -20, "VRRight": 30, "VRBottom": 40}}})
    assert map_bounds(root["m"]) == Bounds(-10, -20, 30, 40)


def test_map_bounds_falls_back_to_footholds() -> None:
    fh = {"1": {"x1": 0, "y1": 100, "x2": 500, "y2": 100}, "2": {"x1": 500, "y1": 100, "x2": 600, "y2": 50}}
    root = _tree({"m": {"info": {}, "foothold": {"0": {"1": fh}}}})
    assert map_bounds(root["m"]) == Bounds(0, 50 - 250, 600, 100 + 200)


def test_blit_clips_and_flips() -> None:
    renderer = MapRenderer(wz=None)  # type: ignore[arg-type]
    canvas = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
    sprite_img = Image.new("RGBA", (2, 1))
    sprite_img.putpixel((0, 0), (255, 0, 0, 255))
    sprite_img.putpixel((1, 0), (0, 255, 0, 255))
    sprite = Sprite(sprite_img, (0, 0), 0)
    ctx = DrawContext(canvas, Bounds(0, 0, 4, 4), 1.0, [])

    renderer._blit(ctx, sprite, -1, 0)
    assert canvas.getpixel((0, 0)) == (0, 255, 0, 255)

    renderer._blit(ctx, sprite, 3, 3, flip=True)
    assert canvas.getpixel((1, 3)) == (0, 255, 0, 255)
    assert canvas.getpixel((2, 3)) == (255, 0, 0, 255)
