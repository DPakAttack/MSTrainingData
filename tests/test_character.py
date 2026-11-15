"""Character sprite API: pure helpers, plus smoke tests against the sample `Data` folder."""

from __future__ import annotations

import base64
import random
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from api import WzApi
from character.build import normalize_weights, resolve_assignments, sample_optional, sample_styled
from character.catalog import ItemInfo, split_slot_codes
from character.codec import CharacterSpec, decode
from character.equip import (
    OPTIONAL_SLOTS,
    REQUIRED_SLOTS,
    EquipType,
    equip_type_of,
    face_style,
    hair_style,
    item_gender,
)
from character.renderer import _Layer, solve_positions

E = EquipType

# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def test_codec_round_trip() -> None:
    spec = CharacterSpec.from_mapping(
        {E.WEAPON: 1702024, E.SKIN: 6, E.FACE: 21025, E.HAIR: 49999, E.CAP: 1009999, E.SHIELD: 1092000}
    )
    name = spec.encode()
    assert name.startswith("c1_")
    assert name == name.lower() and name.isascii() and "=" not in name
    assert decode(name) == spec
    assert decode(f"{name}.png") == spec
    assert decode(Path("out") / "Character" / f"{name}.png") == spec
    assert [slot for slot, _ in spec.parts] == [E.SKIN, E.FACE, E.HAIR, E.CAP, E.SHIELD, E.WEAPON]


def test_codec_is_order_independent() -> None:
    a = CharacterSpec.from_mapping({E.SKIN: 0, E.FACE: 20000, E.HAIR: 30000, E.COAT: 1040002})
    b = CharacterSpec.from_mapping({E.COAT: 1040002, E.HAIR: 30000, E.FACE: 20000, E.SKIN: 0})
    assert a.encode() == b.encode()


@pytest.mark.parametrize("name", ["x1_aaaa", "c1_", "c1_!!!!", "c2_aeaaaaaa"])
def test_codec_rejects_invalid(name: str) -> None:
    with pytest.raises(ValueError):
        decode(name)


def test_codec_rejects_other_version() -> None:
    name = CharacterSpec.from_mapping({E.SKIN: 0, E.FACE: 20000, E.HAIR: 30000}).encode()
    raw = bytearray(base64.b32decode(name[3:].upper() + "=" * (-len(name[3:]) % 8)))
    raw[0] = 2
    forged = "c1_" + base64.b32encode(bytes(raw)).decode().rstrip("=").lower()
    with pytest.raises(ValueError, match="version"):
        decode(forged)


def test_spec_missing_required() -> None:
    assert CharacterSpec.from_mapping({E.SKIN: 0}).missing_required() == [E.FACE, E.HAIR]


# ---------------------------------------------------------------------------
# Id helpers
# ---------------------------------------------------------------------------


def test_equip_type_of() -> None:
    assert equip_type_of(20000) == E.FACE
    assert equip_type_of(30000) == E.HAIR and equip_type_of(40000) == E.HAIR
    assert equip_type_of(1002000) == E.CAP
    assert equip_type_of(1012000) == E.FACE_ACCESSORY
    assert equip_type_of(1022000) == E.EYE_ACCESSORY
    assert equip_type_of(1050000) == E.LONGCOAT
    assert equip_type_of(1092000) == E.SHIELD
    assert equip_type_of(1302000) == E.WEAPON and equip_type_of(1702000) == E.WEAPON
    with pytest.raises(ValueError, match="supported"):
        equip_type_of(1112000)  # ring


def test_gender_and_styles() -> None:
    assert item_gender(30000) == 0 and item_gender(31000) == 1 and item_gender(1002000) == 2
    assert hair_style(30007) == hair_style(30000) != hair_style(30010)
    assert face_style(20700) == face_style(20000) != face_style(20001)


def test_split_slot_codes() -> None:
    assert split_slot_codes("MaPn") == ("Ma", "Pn")
    assert split_slot_codes("CpH1H5") == ("Cp", "H1", "H5")
    assert split_slot_codes(None) == ()


# ---------------------------------------------------------------------------
# Assignment resolution
# ---------------------------------------------------------------------------


def _info(item_id: int, slot: EquipType, islot: str) -> ItemInfo:
    return ItemInfo(item_id, slot, split_slot_codes(islot), split_slot_codes(islot), 2)


_ITEMS = {
    (E.SKIN, 0): _info(0, E.SKIN, "Bd"),
    (E.FACE, 20000): _info(20000, E.FACE, "Fc"),
    (E.HAIR, 30000): _info(30000, E.HAIR, "Hr"),
    (E.COAT, 1040002): _info(1040002, E.COAT, "Ma"),
    (E.PANTS, 1060002): _info(1060002, E.PANTS, "Pn"),
    (E.LONGCOAT, 1050000): _info(1050000, E.LONGCOAT, "MaPn"),
    (E.SHIELD, 1092000): _info(1092000, E.SHIELD, "Si"),
    (E.WEAPON, 1302000): _info(1302000, E.WEAPON, "Wp"),
    (E.WEAPON, 1402000): _info(1402000, E.WEAPON, "WpSi"),
    (E.CAP, 1002999): _info(1002999, E.CAP, "HrCp"),
    (E.HAIR, 30010): _info(30010, E.HAIR, "Hr"),
}
_BASE = [(E.SKIN, 0), (E.FACE, 20000), (E.HAIR, 30000)]


def _resolve(*extra: tuple[EquipType, int]) -> dict[str, int]:
    return resolve_assignments([*_BASE, *extra], lambda s, i: _ITEMS.get((s, i))).to_dict()


def test_hair_claiming_cap_keeps_hair() -> None:
    out = _resolve((E.CAP, 1002999))
    assert out["hair"] == 30000 and out["cap"] == 1002999
    out = _resolve((E.CAP, 1002999), (E.HAIR, 30010))
    assert out["hair"] == 30010 and out["cap"] == 1002999


def test_longcoat_evicts_coat_and_pants() -> None:
    out = _resolve((E.COAT, 1040002), (E.PANTS, 1060002), (E.LONGCOAT, 1050000))
    assert out == {"skin": 0, "face": 20000, "hair": 30000, "longcoat": 1050000}


def test_coat_evicts_longcoat() -> None:
    out = _resolve((E.LONGCOAT, 1050000), (E.COAT, 1040002))
    assert "longcoat" not in out and out["coat"] == 1040002


def test_two_handed_weapon_and_shield_latest_wins() -> None:
    assert "shield" not in _resolve((E.SHIELD, 1092000), (E.WEAPON, 1402000))
    out = _resolve((E.WEAPON, 1402000), (E.SHIELD, 1092000))
    assert "weapon" not in out and out["shield"] == 1092000
    assert _resolve((E.SHIELD, 1092000), (E.WEAPON, 1302000)).keys() >= {"shield", "weapon"}


def test_same_slot_replaced() -> None:
    assert _resolve((E.WEAPON, 1302000), (E.WEAPON, 1402000))["weapon"] == 1402000


def test_unknown_item_raises() -> None:
    with pytest.raises(ValueError, match="not a drawable"):
        _resolve((E.CAP, 1002000))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_styled_is_uniform_over_styles() -> None:
    # Style 3000 has 8 colors, style 3001 has one; both styles should be picked about equally.
    ids = [30000 + c for c in range(8)] + [30010]
    rng = random.Random(0)
    picks = [sample_styled(rng, ids, hair_style) for _ in range(2000)]
    share = sum(p == 30010 for p in picks) / len(picks)
    assert 0.4 < share < 0.6
    assert set(picks) == set(ids)
    with pytest.raises(ValueError):
        sample_styled(rng, [], hair_style)


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_sample_optional_extreme_weights(weight: float) -> None:
    pools = {slot: [i + 1 for i in range(5)] for slot in OPTIONAL_SLOTS}
    weights = normalize_weights(None, weight)
    for seed in range(20):
        filled = {slot for slot, _ in sample_optional(random.Random(seed), pools, weights)}
        assert filled == (set(OPTIONAL_SLOTS) if weight else set())


def test_sample_optional_weight_does_not_reshuffle_other_slots() -> None:
    pools = {slot: list(range(100)) for slot in OPTIONAL_SLOTS}
    base = dict(sample_optional(random.Random(3), pools, normalize_weights(None, 1.0)))
    tweaked = dict(sample_optional(random.Random(3), pools, normalize_weights({E.CAP: 0.0}, 1.0)))
    assert E.CAP not in tweaked
    assert tweaked == {k: v for k, v in base.items() if k != E.CAP}


def test_normalize_weights() -> None:
    weights = normalize_weights({"cap": 0.3, "SHIELD": 0.05, E.CAPE: 1}, 0.5)
    assert weights[E.CAP] == 0.3 and weights[E.SHIELD] == 0.05 and weights[E.CAPE] == 1.0
    assert weights[E.COAT] == 0.5 and set(weights) == set(OPTIONAL_SLOTS)
    for bad in ({E.HAIR: 0.5}, {"petequip": 0.5}, {E.CAP: 1.5}):
        with pytest.raises(ValueError):
            normalize_weights(bad, 0.5)


# ---------------------------------------------------------------------------
# Anchor solver
# ---------------------------------------------------------------------------


def _layer(name: str, anchors: dict[str, tuple[int, int]]) -> _Layer:
    return _Layer(E.SKIN, name, Image.new("RGBA", (1, 1)), (0, 0), anchors, "body", 0)


def test_solve_positions_chains_anchors() -> None:
    body = _layer("body", {"navel": (-8, -21), "neck": (-4, -32)})
    head = _layer("head", {"neck": (0, 15), "brow": (-4, -5)})
    face = _layer("face", {"brow": (-1, -3)})
    loose = _layer("loose", {"nothing": (1, 1)})
    # The face is listed before the head that defines its anchor, so this needs a second pass.
    assert solve_positions([body, face, head, loose]) == [(0, 0), (-7, -49), (-4, -47), (0, 0)]


# ---------------------------------------------------------------------------
# Smoke tests against Data/
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[WzApi]:
    cache = tmp_path_factory.getbasetemp() / "character_catalog.json"
    out = tmp_path_factory.mktemp("character_out")
    with WzApi(data_dir, out, character_catalog_cache=cache) as wz_api:
        yield wz_api


def _bare(api: WzApi) -> list[tuple[EquipType | None, int]]:
    char = api.character
    return [(E.SKIN, char.ids(E.SKIN)[0]), (E.FACE, char.ids(E.FACE)[0]), (E.HAIR, char.ids(E.HAIR)[0])]


def test_pools_non_empty(api: WzApi) -> None:
    assert len(api.character.ids(E.SKIN)) >= 1
    for slot in (*REQUIRED_SLOTS, *OPTIONAL_SLOTS):
        assert api.character.ids(slot), slot
        for item_id in api.character.ids(slot)[:3]:
            assert slot == E.SKIN or equip_type_of(item_id) == slot


def test_render_is_deterministic(api: WzApi) -> None:
    spec = api.character.random(seed=11, weight_map=None)
    a = api.character.render(spec)
    b = api.character.render(spec.encode())
    assert a.image.mode == "RGBA" and a.image.width > 10 and a.image.height > 10
    assert a.origin == b.origin and a.image.tobytes() == b.image.tobytes()


def test_cap_hides_front_hair(api: WzApi) -> None:
    char = api.character
    catalog = char.renderer.catalog
    cap = next(i for i in char.ids(E.CAP) if "H1" in catalog.get(E.CAP, i).vslot)
    hair_layer = _Layer(E.HAIR, "hairOverHead", Image.new("RGBA", (1, 1)), (0, 0), {}, "hairOverHead", 0)
    assert "H1" in char.renderer.smap["hairOverHead"]
    assert char.renderer._visible(hair_layer, {"H1": E.HAIR})
    assert not char.renderer._visible(hair_layer, {"H1": E.CAP})

    bare = char.render(char.build(_bare(api)))
    capped = char.render(char.build([*_bare(api), (None, cap)]))
    assert bare.image.size != capped.image.size or ImageChops.difference(bare.image, capped.image).getbbox()


def test_random_is_reproducible(api: WzApi) -> None:
    char = api.character
    assert char.random(seed=1) == char.random(seed=1)
    specs = {char.random(seed=s) for s in range(20)}
    assert len(specs) > 15
    for spec in specs:
        assert not spec.missing_required()


def test_random_weight_extremes(api: WzApi) -> None:
    char = api.character
    none = char.random(seed=4, weight_map=dict.fromkeys(OPTIONAL_SLOTS, 0.0))
    assert set(none) == set(REQUIRED_SLOTS)
    full = char.random(seed=4, weight_map=dict.fromkeys(OPTIONAL_SLOTS, 1.0))
    # Conflicts (overall vs top/pants, two-handed weapon vs shield) can evict some filled slots.
    assert len(full) >= len(REQUIRED_SLOTS) + len(OPTIONAL_SLOTS) - 3


def test_random_consistent_gender(api: WzApi) -> None:
    for seed in range(10):
        spec = api.character.random(seed=seed, weight_map=dict.fromkeys(OPTIONAL_SLOTS, 1.0))
        genders = {item_gender(i) for slot, i in spec.items() if slot != E.SKIN} - {2}
        assert len(genders) <= 1, spec.to_dict()


def test_build_latest_wins_against_data(api: WzApi) -> None:
    char = api.character
    catalog = char.renderer.catalog
    longcoat = next(i for i in char.ids(E.LONGCOAT) if {"Ma", "Pn"} <= set(catalog.get(E.LONGCOAT, i).islot))
    coat, pants = char.ids(E.COAT)[0], char.ids(E.PANTS)[0]
    spec = char.build([*_bare(api), (None, coat), (None, pants), (None, longcoat)])
    assert E.COAT not in spec and E.PANTS not in spec and spec[E.LONGCOAT] == longcoat
    with pytest.raises(ValueError, match="Missing required"):
        char.build([(None, coat)])


def test_dump_round_trips_name(api: WzApi) -> None:
    char = api.character
    spec = char.random(seed=2)
    [path] = char.dump([spec])
    assert path.parent.name == "Character" and path.suffix == ".png"
    assert decode(path.name) == spec
    with Image.open(path) as image:
        assert image.mode == "RGBA"
        x, y = (int(v) for v in image.text["origin"].split(","))
        assert 0 <= x <= image.width and 0 <= y <= image.height
    rng = random.Random(9)
    expected = [char.random(rng) for _ in range(3)]
    assert [decode(p) for p in char.dump_random(3, seed=9)] == expected
