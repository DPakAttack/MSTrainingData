"""End-to-end smoke tests against the real `Data` folder (skipped when it is not present)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api import WzApi
from main import main


@pytest.fixture(scope="module")
def api(data_dir: Path) -> WzApi:
    with WzApi(data_dir) as wz:
        yield wz


def test_structure_top_level(api: WzApi) -> None:
    names = {c.name for c in api.wz.root}
    assert {"Map", "Mob", "String"} <= names


def test_mob_list(api: WzApi) -> None:
    mobs = api.mob.list()
    assert (2, "Snail") in mobs
    assert mobs == sorted(mobs)


def test_map_list(api: WzApi) -> None:
    maps = dict(api.map.list())
    assert 10 in maps
    assert maps[10] and "Mushroom Town" in maps[10]


def test_mob_dump(api: WzApi, tmp_path: Path) -> None:
    (out,) = api.mob.dump([2], output_root=tmp_path)
    assert out == tmp_path / "Mob" / "Snail - 2"
    info = json.loads((out / "info.json").read_text(encoding="utf-8"))
    assert info["id"] == 2 and info["image_count"] > 0
    frame = info["tree"]["move"]["0"]
    assert (out / frame["_image"]).is_file()
    assert set(frame["origin"]) == {"x", "y"}


def test_map_dump_is_deterministic(api: WzApi, tmp_path: Path) -> None:
    (first,) = api.map.dump([10], output_root=tmp_path / "a")
    (second,) = api.map.dump([10], output_root=tmp_path / "b")
    for name in ("map.png", "minimap.png", "info.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    info = json.loads((first / "info.json").read_text(encoding="utf-8"))
    b = info["bounds"]
    assert info["image_size"] == [b["right"] - b["left"], b["bottom"] - b["top"]]
    portals = {p["name"]: p for p in info["portals"]}
    assert portals["in00"]["target_map"] == 1 and portals["in00"]["bidirectional"] is True
    assert portals["sp"]["spawn_point"]
    assert info["footholds"] and all(f["kind"] in ("floor", "wall") for f in info["footholds"])


def test_map_render_draws_portals(api: WzApi) -> None:
    node = api.map.node(10)
    renderer = api.map.renderer
    with_portals = renderer.render(node).image
    renderer.include_portals = False
    try:
        without_portals = renderer.render(node).image
    finally:
        renderer.include_portals = True
    assert with_portals.tobytes() != without_portals.tobytes()


def test_cli_mob_list(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([f"--root_wz={data_dir}", "mob", "list"]) == 0
    assert "(2, 'Snail')" in capsys.readouterr().out


def test_cli_dump_requires_output_root(data_dir: Path) -> None:
    assert main([f"--root_wz={data_dir}", "map", "dump", "--ids", "10"]) == 2
