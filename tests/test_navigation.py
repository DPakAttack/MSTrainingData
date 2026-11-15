"""Unit tests for the normalized foothold, ladder, and portal encoding."""

from __future__ import annotations

import navigation
from wz.node import WzNode


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


def _portal(pn: str, pt: int, tm: int, tn: str, **extra: object) -> dict:
    return {"pn": pn, "pt": pt, "x": 0, "y": 0, "tm": tm, "tn": tn, **extra}


def test_footholds_and_ladders() -> None:
    root = _tree(
        {
            "foothold": {
                "0": {
                    "1": {
                        "2": {"x1": 10, "y1": 0, "x2": 10, "y2": 50, "prev": 1, "next": 0, "piece": 0},
                        "1": {"x1": 0, "y1": 0, "x2": 10, "y2": 0, "prev": 0, "next": 2, "forbidFallDown": 1},
                    }
                }
            },
            "ladderRope": {"1": {"l": 0, "uf": 1, "x": 5, "y1": 40, "y2": -20, "page": 0}},
        }
    )
    fhs = navigation.footholds(root)
    assert [f["id"] for f in fhs] == [1, 2]
    assert fhs[0]["kind"] == "floor" and not fhs[0]["can_drop_through"] and fhs[0]["next"] == 2
    assert fhs[1]["kind"] == "wall" and fhs[1]["can_drop_through"] and fhs[1]["next"] is None
    (rope,) = navigation.ladders(root)
    assert rope == {
        "id": 1, "kind": "rope", "x": 5, "top": -20, "bottom": 40, "top_exit": True, "page": 0, "extra": {}
    }  # fmt: skip


def test_portal_directionality() -> None:
    here = _tree(
        {
            "portal": {
                "0": _portal("sp", 0, navigation.NO_MAP, ""),
                "1": _portal("two_way", 2, 20, "back"),
                "2": _portal("one_way", 2, 30, "drop"),
                "3": _portal("missing", 2, 40, "x"),
                "4": _portal("warp_a", 1, 10, "warp_b"),
                "5": _portal("warp_b", 1, 10, "elsewhere"),
                "6": _portal("scripted", 7, navigation.NO_MAP, "", script="go"),
            }
        }
    )
    tables = {
        20: {"back": (10, "two_way")},
        30: {"drop": (None, None), "other": (10, "one_way")},
    }
    out = {p["name"]: p for p in navigation.portals(here, 10, tables.get)}

    assert out["sp"]["spawn_point"] and not out["sp"]["usable"] and out["sp"]["bidirectional"] is None
    assert out["two_way"]["bidirectional"] is True and out["two_way"]["returns_to"] == "two_way"
    assert out["one_way"]["bidirectional"] is False and out["one_way"]["target_map_returns"] is True
    assert out["missing"]["target_found"] is False and out["missing"]["bidirectional"] is None
    assert out["warp_a"]["same_map"] and out["warp_a"]["bidirectional"] is True
    assert out["warp_a"]["target_map_returns"] is None
    assert out["scripted"]["usable"] and out["scripted"]["target_map"] is None
