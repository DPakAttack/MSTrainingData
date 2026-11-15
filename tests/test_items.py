"""Item API smoke tests against the sample `Data` folder."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

import main
from api import WzApi
from items import ItemCategory, drop_frames, inventory_icon

POTION, CAP, PET, MESO, MESO_10 = 2000000, 1002000, 5000000, 9000000, 9000005


@pytest.fixture(scope="module")
def api(data_dir: Path) -> Iterator[WzApi]:
    with WzApi(data_dir) as wz_api:
        yield wz_api


def test_every_category_non_empty(api: WzApi) -> None:
    for category in ItemCategory:
        assert api.item.ids(category), category


def test_ids_unique_and_sorted(api: WzApi) -> None:
    ids = api.item.ids()
    assert ids == sorted(set(ids))
    by_category = [i for c in ItemCategory for i in api.item.ids(c)]
    assert sorted(by_category) == ids
    assert api.item.ids([ItemCategory.PET, ItemCategory.SPECIAL]) == sorted(
        api.item.ids(ItemCategory.PET) + api.item.ids(ItemCategory.SPECIAL)
    )


def test_names_and_categories(api: WzApi) -> None:
    assert api.item.name(POTION) == "Red Potion"
    assert api.item.name(CAP) == "Brown Skullcap"
    assert api.item.name(PET) == "Brown Kitty"
    assert api.item.category(POTION) == ItemCategory.CONSUME
    assert api.item.category(CAP) == ItemCategory.EQUIP
    assert api.item.category(PET) == ItemCategory.PET
    assert api.item.category(MESO) == ItemCategory.SPECIAL
    assert api.item.category(1) is None
    assert dict(api.item.list(ItemCategory.CONSUME))[POTION] == "Red Potion"
    assert all(api.item.category(i) == ItemCategory.PET for i, _ in api.item.list(ItemCategory.PET))


def test_equipment_excludes_faces_and_hair(api: WzApi) -> None:
    equip = api.item.ids(ItemCategory.EQUIP)
    assert not any(20000 <= i < 50000 for i in equip)


def test_drop_frames(api: WzApi) -> None:
    meso = drop_frames(api.item.node(MESO))
    assert [f.name for f in meso] == ["0", "1", "2", "3"]
    meso10 = drop_frames(api.item.node(MESO_10))
    assert [f.name for f in meso10] == [str(i) for i in range(10)]
    assert all(f.get_int("delay") == 60 for f in meso10)
    [potion] = drop_frames(api.item.node(POTION))
    assert potion.path.endswith("info/iconRaw")
    assert inventory_icon(api.item.node(POTION)).path.endswith("info/icon")
    assert inventory_icon(api.item.node(MESO)) is None
    # Special packs have only `icon`, which is then used for the drop sprite too.
    [pack] = drop_frames(api.item.node(9100000))
    assert pack.name == "icon"


@pytest.mark.parametrize(
    ("item_id", "folder", "frames", "has_icon"),
    [
        (POTION, "Consume/Red Potion - 2000000", 1, True),
        (CAP, "Equip/Brown Skullcap - 1002000", 1, True),
        (PET, "Pet/Brown Kitty - 5000000", 1, True),
        (MESO, "Special/unknown - 9000000", 4, False),
        (MESO_10, "Special/unknown - 9000005", 10, False),
    ],
)
def test_dump(api: WzApi, tmp_path: Path, item_id: int, folder: str, frames: int, has_icon: bool) -> None:
    [out] = api.item.dump([item_id], output_root=tmp_path)
    assert out == tmp_path / "Item" / Path(folder)
    info = json.loads((out / "info.json").read_text(encoding="utf-8"))
    assert info["id"] == item_id and info["category"] == out.parent.name
    assert len(info["drop"]) == frames
    for frame in info["drop"]:
        with Image.open(out / frame["image"]) as image:
            assert image.mode == "RGBA" and list(image.size) == frame["size"]
        assert frame["delay"] > 0 and len(frame["origin"]) == 2
    assert (info["icon"] == "icon.png") == has_icon == (out / "icon.png").is_file()
    assert not any("icon" in key.lower() for key in info["info"])


def test_dump_reports_info_values(api: WzApi, tmp_path: Path) -> None:
    [out] = api.item.dump([CAP], output_root=tmp_path)
    info = json.loads((out / "info.json").read_text(encoding="utf-8"))
    assert info["info"]["islot"] == "Cp" and info["info"]["reqLevel"] == 5
    [out] = api.item.dump([MESO_10], output_root=tmp_path)
    assert {f["delay"] for f in json.loads((out / "info.json").read_text())["drop"]} == {60}
    [out] = api.item.dump([MESO], output_root=tmp_path)
    assert {f["delay"] for f in json.loads((out / "info.json").read_text())["drop"]} == {120}


def test_dump_category_filter_and_unknown_ids(api: WzApi, tmp_path: Path) -> None:
    written = api.item.dump([POTION, CAP, 1], categories=[ItemCategory.EQUIP], output_root=tmp_path)
    assert [p.name for p in written] == ["Brown Skullcap - 1002000"]


def test_cli(api: WzApi, data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main.main(["--root_wz", str(data_dir), "item", "list", "--category", "pet", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {"id": PET, "name": "Brown Kitty"} in listed
    assert [e["id"] for e in listed] == api.item.ids(ItemCategory.PET)
    args = ["--root_wz", str(data_dir), "--output_root", str(tmp_path), "item", "dump"]
    assert main.main([*args, "--category", "Special", "Consume", "--ids", str(POTION), str(CAP)]) == 0
    assert capsys.readouterr().out.strip().splitlines() == [str(tmp_path / "Item" / "Consume" / "Red Potion - 2000000")]
    with pytest.raises(SystemExit):
        main.main([*args, "--category", "bogus"])
