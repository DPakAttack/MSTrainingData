"""Normalized walkable geometry and portal links of a map, for `info.json`.

Coordinates are world pixels (y grows downward), the same space as `bounds` in `info.json`.

- Footholds are line segments characters stand on. Walking along them goes both ways; chains are
  linked through `prev`/`next`. Vertical segments are walls. Characters can jump up through any
  floor; dropping down through one is allowed unless `can_drop_through` is false.
- Ladders/ropes are vertical segments climbable both ways between `top` and `bottom`;
  `top_exit` says whether the top connects to a platform the character can step onto.
- Portals are directed edges `(this map, name) -> (target_map, target_portal)`. A link is two-way
  only when the target portal leads back to this map (`bidirectional`), which is derived here by
  looking the target portal up. Spawn points (`pt=0`) are arrival points only, and script portals
  pick their destination at runtime, so their target is not known statically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from render.map_renderer import HIDDEN_PORTAL_TYPES, PORTAL_TYPES
from wz.node import WzNode

NO_MAP = 999999999
"""`tm` value meaning the portal has no static destination."""

PortalTable = dict[str, tuple[int | None, str | None]]
"""`portal name -> (target_map, target_portal)` for one map."""

_FOOTHOLD_KEYS = frozenset({"x1", "y1", "x2", "y2", "prev", "next", "piece", "forbidFallDown"})
_LADDER_KEYS = frozenset({"x", "y1", "y2", "l", "uf", "page", "piece"})
_PORTAL_KEYS = frozenset({"pn", "pt", "x", "y", "tm", "tn", "script", "onlyOnce", "hideTooltip", "delay", "image"})


def _extra(node: WzNode, known: frozenset[str]) -> dict[str, Any]:
    return {c.name: c.value for c in node if c.name not in known and len(c) == 0}


def _target(portal: WzNode) -> tuple[int | None, str | None]:
    tm = portal.get_int("tm", NO_MAP)
    return (None if tm == NO_MAP else tm), (portal.get_str("tn") or None)


def portal_table(map_node: WzNode) -> PortalTable:
    """Collect `portal name -> (target_map, target_portal)` for `map_node`."""
    table: PortalTable = {}
    for portal in map_node.get("portal") or ():
        name = portal.get_str("pn")
        if name and name not in table:
            table[name] = _target(portal)
    return table


def footholds(map_node: WzNode) -> list[dict[str, Any]]:
    """Every foothold segment, ordered by `(layer, group, id)`."""
    out: list[dict[str, Any]] = []
    for layer in sorted(map_node.get("foothold") or (), key=lambda n: int(n.name)):
        for group in sorted(layer, key=lambda n: int(n.name)):
            for fh in sorted(group, key=lambda n: int(n.name)):
                x1, y1, x2, y2 = (fh.get_int(k) for k in ("x1", "y1", "x2", "y2"))
                out.append(
                    {
                        "id": int(fh.name),
                        "layer": int(layer.name),
                        "group": int(group.name),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "kind": "wall" if x1 == x2 else "floor",
                        "prev": fh.get_int("prev") or None,
                        "next": fh.get_int("next") or None,
                        "piece": fh.get_int("piece"),
                        "can_drop_through": not fh.get_int("forbidFallDown"),
                        "extra": _extra(fh, _FOOTHOLD_KEYS),
                    }
                )
    return out


def ladders(map_node: WzNode) -> list[dict[str, Any]]:
    """Every ladder and rope, ordered by id."""
    out: list[dict[str, Any]] = []
    for lr in sorted(map_node.get("ladderRope") or (), key=lambda n: int(n.name)):
        y1, y2 = lr.get_int("y1"), lr.get_int("y2")
        out.append(
            {
                "id": int(lr.name),
                "kind": "ladder" if lr.get_int("l") else "rope",
                "x": lr.get_int("x"),
                "top": min(y1, y2),
                "bottom": max(y1, y2),
                "top_exit": bool(lr.get_int("uf")),
                "page": lr.get_int("page"),
                "extra": _extra(lr, _LADDER_KEYS),
            }
        )
    return out


def portals(map_node: WzNode, map_id: int, lookup: Callable[[int], PortalTable | None]) -> list[dict[str, Any]]:
    """Every portal as a directed edge, with its reverse link resolved.

    Parameters
    ----------
    map_node
        The map whose portals to encode.
    map_id
        Id of `map_node` (used to detect links leading back here).
    lookup
        Returns the `PortalTable` of another map id, or `None` if that map is not in the data.

    Returns
    -------
    One dict per portal, ordered by index. `bidirectional` is `True` when the target portal leads
    back to this map, `False` when it leads elsewhere or nowhere, and `None` when the destination
    is unknown (no static target, or the target map/portal is missing from the data).
    `target_map_returns` is the looser check that any portal of the target map leads here
    (`None` for same-map portals or unknown targets).
    """
    own = portal_table(map_node)
    out: list[dict[str, Any]] = []
    for portal in sorted(map_node.get("portal") or (), key=lambda n: int(n.name)):
        pt = portal.get_int("pt")
        name = portal.get_str("pn")
        target_map, target_portal = _target(portal)
        script = portal.get_str("script") or None

        target_found: bool | None = None
        returns_to: str | None = None
        bidirectional: bool | None = None
        target_map_returns: bool | None = None
        if target_map is not None:
            table = own if target_map == map_id else lookup(target_map)
            back = table.get(target_portal) if table is not None and target_portal else None
            target_found = back is not None
            if back is not None:
                bidirectional = back[0] == map_id
                returns_to = back[1] if bidirectional else None
            if table is not None and target_map != map_id:
                target_map_returns = any(tm == map_id for tm, _ in table.values())

        out.append(
            {
                "index": int(portal.name),
                "name": name,
                "type": pt,
                "type_name": PORTAL_TYPES[pt] if 0 <= pt < len(PORTAL_TYPES) else None,
                "x": portal.get_int("x"),
                "y": portal.get_int("y"),
                "spawn_point": pt == 0,
                "hidden": pt in HIDDEN_PORTAL_TYPES,
                "usable": pt != 0 and (target_map is not None or script is not None),
                "target_map": target_map,
                "target_portal": target_portal,
                "same_map": target_map == map_id,
                "script": script,
                "only_once": bool(portal.get_int("onlyOnce")),
                "target_found": target_found,
                "bidirectional": bidirectional,
                "returns_to": returns_to,
                "target_map_returns": target_map_returns,
                "extra": _extra(portal, _PORTAL_KEYS),
            }
        )
    return out
