"""Scene generation: config, decks, box math, and smoke tests against the sample `Data` folder."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import main
from api import WzApi
from render.map_renderer import Bounds
from scene.config import CountSpec, parse_config
from scene.deck import Deck
from scene.generator import box_stats, flip_sprite
from scene.sprites import MESO_IDS, EntitySprite, mob_state

# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------


def test_deck_draws_each_card_once_per_cycle() -> None:
    deck = Deck(list(range(10)), random.Random(0))
    for cycle in range(1, 4):
        drawn = [deck.draw() for _ in range(10)]
        assert sorted(drawn) == list(range(10))
        assert deck.cycles == cycle and deck.remaining == 0
    with pytest.raises(ValueError):
        Deck([], random.Random(0))


def test_deck_is_shuffled_and_seeded() -> None:
    a, b = Deck(list(range(50)), random.Random(1)), Deck(list(range(50)), random.Random(1))
    first = [a.draw() for _ in range(120)]
    assert first == [b.draw() for _ in range(120)]
    assert first[:50] != list(range(49, -1, -1)) and first[:50] != first[50:100]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults_and_round_trip() -> None:
    cfg = parse_config('{"num_samples": 3, "mob": {"count": 2}}')
    assert cfg.resolution == (1366, 768) and cfg.seed is None
    assert cfg.mobs == CountSpec(2, 2) and cfg.characters == CountSpec(0, 0) and cfg.items == CountSpec(0, 0)
    assert cfg.meso_fraction == 0.5 and cfg.flip_probability == 0.5
    full = parse_config(
        {
            "num_samples": 2,
            "resolution": [640, 480],
            "seed": 4,
            "character": {"min_count": 1, "max_count": 3, "equip_weights": {"cap": 0.3}},
            "item": {"min_count": 0, "max_count": 5, "meso_fraction": 0.25},
            "maps": [10000],
        }
    )
    assert full.resolution == (640, 480) and full.characters == CountSpec(1, 3) and full.maps == (10000,)
    assert full.equip_weights == {"cap": 0.3} and full.meso_fraction == 0.25
    assert parse_config(full.to_json()) == full
    assert parse_config({"num_samples": 1, "resolution": "800 x 600"}).resolution == (800, 600)


@pytest.mark.parametrize(
    "config",
    [
        "not json",
        {},
        {"num_samples": 0},
        {"num_samples": 1, "mob": {"count": 1, "min_count": 0, "max_count": 2}},
        {"num_samples": 1, "mob": {}},
        {"num_samples": 1, "mob": {"min_count": 1}},
        {"num_samples": 1, "mob": {"min_count": 3, "max_count": 1}},
        {"num_samples": 1, "mob": {"count": -1}},
        {"num_samples": 1, "mob": {"count": 1, "speed": 2}},
        {"num_samples": 1, "resolution": "big"},
        {"num_samples": 1, "resolution": [0, 10]},
        {"num_samples": 1, "character": {"count": 1, "equip_weights": {"HAIR": 0.5}}},
        {"num_samples": 1, "character": {"count": 1, "equip_weights": {"PETEQUIP": 0.5}}},
        {"num_samples": 1, "character": {"count": 1, "equip_weights": {"CAP": 2}}},
        {"num_samples": 1, "item": {"count": 1, "meso_fraction": 1.5}},
        {"num_samples": 1, "flip_probability": -0.1},
        {"num_samples": 1, "maps": []},
        {"num_samples": 1, "unknown": 1},
    ],
)
def test_config_rejects_invalid(config: object) -> None:
    with pytest.raises(ValueError):
        parse_config(config)  # type: ignore[arg-type]


def test_count_spec_sampling_is_uniform_and_inclusive() -> None:
    rng = random.Random(0)
    counts = Counter(CountSpec(1, 3).sample(rng) for _ in range(3000))
    assert set(counts) == {1, 2, 3} and all(800 < v < 1200 for v in counts.values())


# ---------------------------------------------------------------------------
# Boxes and sprites
# ---------------------------------------------------------------------------


def test_box_stats_visible_and_full() -> None:
    alpha = np.zeros((10, 10), np.uint8)
    alpha[2:8, 3:9] = 255  # 36 opaque pixels, sprite box [3, 2, 9, 8] relative
    owner = np.zeros((20, 30), np.int32)
    owner[7:13, 13:19] = 5  # entity drawn at (10, 5)
    owner[7:13, 16:19] = 0  # right half hidden by scenery
    stats = box_stats(alpha, (10, 5), owner, 5)
    assert stats["full_bbox"] == [13, 7, 19, 13]
    assert stats["bbox"] == [13, 7, 16, 13]
    assert stats["visible_fraction"] == 0.5 and stats["truncated"] is False


def test_box_stats_truncated_and_hidden() -> None:
    alpha = np.full((4, 4), 255, np.uint8)
    owner = np.zeros((10, 10), np.int32)
    owner[0:2, 0:2] = 1
    stats = box_stats(alpha, (-2, -2), owner, 1)
    assert stats["truncated"] and stats["full_bbox"] == [0, 0, 2, 2] and stats["bbox"] == [0, 0, 2, 2]
    assert stats["visible_fraction"] == 0.25
    hidden = box_stats(alpha, (3, 3), np.zeros((10, 10), np.int32), 1)
    assert hidden["bbox"] is None and hidden["visible_fraction"] == 0.0 and hidden["full_bbox"] == [3, 3, 7, 7]
    off = box_stats(alpha, (50, 50), owner, 1)
    assert off["full_bbox"] is None and off["bbox"] is None and off["truncated"]


def test_flip_sprite_mirrors_origin() -> None:
    image = Image.new("RGBA", (10, 4))
    image.putpixel((1, 0), (255, 0, 0, 255))
    flipped = flip_sprite(EntitySprite(image, (3, 4)))
    assert flipped.origin == (7, 4) and flipped.image.getpixel((8, 0)) == (255, 0, 0, 255)


def test_mob_state() -> None:
    assert [mob_state(a) for a in ("hit1", "die1", "attack2", "stand", "skill12", "fly")] == [
        "hit",
        "die",
        "attack",
        "stand",
        "skill",
        "fly",
    ]


# ---------------------------------------------------------------------------
# Smoke tests against Data/
# ---------------------------------------------------------------------------

_CONFIG = {
    "num_samples": 3,
    "seed": 11,
    "resolution": "640x400",
    "character": {"min_count": 1, "max_count": 2, "equip_weights": {"CAP": 0.5, "SHIELD": 0.1}},
    "mob": {"min_count": 1, "max_count": 4},
    "item": {"count": 4},
}


@pytest.fixture(scope="module")
def api(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[WzApi]:
    cache = tmp_path_factory.getbasetemp() / "character_catalog.json"
    with WzApi(data_dir, tmp_path_factory.mktemp("scenes"), character_catalog_cache=cache) as wz_api:
        yield wz_api


@pytest.fixture(scope="module")
def run_dir(api: WzApi) -> Path:
    return api.generate_scene(json.dumps(_CONFIG))


def _metadata(run: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run.glob("*.metadata.json"))]


def test_outputs_written(run_dir: Path) -> None:
    assert run_dir.parent.name == "Scenes"
    config = json.loads((run_dir / "config.json").read_text())
    assert config["seed"] == 11 and config["num_samples"] == 3
    metas = _metadata(run_dir)
    assert len(metas) == 3
    for meta in metas:
        sid = meta["sample_id"]
        for name in (f"{sid}.jpg", f"{sid}.bbox.jpg"):
            with Image.open(run_dir / name) as image:
                assert image.size == (640, 400) and image.format == "JPEG"
        assert meta["image"] == f"{sid}.jpg" and meta["bbox_image"] == f"{sid}.bbox.jpg"
        assert isinstance(meta["map"]["id"], int) and "name" in meta["map"]


def test_counts_and_boxes(run_dir: Path) -> None:
    for meta in _metadata(run_dir):
        assert 1 <= len(meta["characters"]) <= 2
        assert 1 <= len(meta["mobs"]) <= 4
        assert len(meta["items"]) == 4
        entries = meta["characters"] + meta["mobs"] + meta["items"]
        assert sorted(e["instance_id"] for e in entries) == list(range(1, len(entries) + 1))
        for e in entries:
            for key in ("bbox", "full_bbox"):
                box = e[key]
                if box is not None:
                    assert 0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 400
            if e["bbox"] is not None:
                fb = e["full_bbox"]
                assert fb[0] <= e["bbox"][0] and fb[1] <= e["bbox"][1]
                assert e["bbox"][2] <= fb[2] and e["bbox"][3] <= fb[3]
            assert 0.0 <= e["visible_fraction"] <= 1.0 and 0 <= e["layer"] <= 7
        for mob in meta["mobs"]:
            assert mob["mob_name"] and mob["state"] == mob_state(mob["action"])
        for item in meta["items"]:
            assert item["is_meso"] == (item["item_id"] in MESO_IDS)
        for char in meta["characters"]:
            assert char["code"].startswith("c1_") and {"skin", "face", "hair"} <= set(char["spec"])


def test_same_seed_is_reproducible(api: WzApi, run_dir: Path) -> None:
    again = api.generate_scene(_CONFIG)
    assert again != run_dir
    assert _metadata(again) == _metadata(run_dir)


def test_owner_map_reports_occlusion(api: WzApi) -> None:
    gen = api._scenes
    assert gen is not None
    map_id = 10001001  # Henesys Townstreet: houses and fences on upper layers
    node = api.map.node(map_id)
    info = gen._info(map_id)
    vp = gen._viewport(info, (800, 600), random.Random(0))
    square = Image.new("RGBA", (vp.width, vp.height), (255, 0, 0, 255))
    fractions = {}
    for layer in (0, 7):

        def hook(current: int, ctx, layer: int = layer) -> None:
            if current == layer:
                gen.renderer.draw_entity(ctx, square, vp.left, vp.top, 1)

        result = gen.renderer.render(node, vp, layer_hook=hook, track_owner=True)
        alpha = np.asarray(square.getchannel("A"))
        fractions[layer] = box_stats(alpha, (0, 0), np.asarray(result.owner), 1)["visible_fraction"]
    node.unload()
    assert fractions[0] < 1.0
    assert fractions[0] <= fractions[7]


def test_viewport_inside_map_when_large_enough(api: WzApi) -> None:
    gen = api._scenes
    info = gen._info(10001001)
    rng = random.Random(3)
    b = info.bounds
    for _ in range(20):
        vp = gen._viewport(info, (640, 400), rng)
        assert (vp.width, vp.height) == (640, 400)
        if b.width >= 640:
            assert b.left <= vp.left and vp.right <= b.right
        if b.height >= 400:
            assert b.top <= vp.top and vp.bottom <= b.bottom
    assert isinstance(vp, Bounds)


def test_cli(data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "scene.json"
    config.write_text(json.dumps({"num_samples": 1, "seed": 1, "resolution": "320x240", "mob": {"count": 1}}))
    args = ["--root_wz", str(data_dir), "--output_root", str(tmp_path), "generate_scene"]
    assert main.main([*args, "--config", f"@{config}"]) == 0
    run = Path(capsys.readouterr().out.strip())
    assert len(list(run.glob("*.metadata.json"))) == 1
    assert main.main([*args, "--config", '{"num_samples": 0}']) == 2
    assert main.main(["--root_wz", str(data_dir), "generate_scene", "--config", "{}"]) == 2
