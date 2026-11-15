"""Command-line interface for the WZ extractor.

Examples
--------
python main.py --root_wz=Data mob list
python main.py --root_wz=Data --output_root=out mob dump --ids 1 2
python main.py --root_wz=Data --output_root=out map dump --ids 10
python main.py --root_wz=Data ls String/Mob.img/2
python main.py --root_wz=Data item list --category Equip --json
python main.py --root_wz=Data --output_root=out item dump --category Special
python main.py --root_wz=Data --output_root=out item dump --ids 2000000 1002000
python main.py --root_wz=Data --output_root=out character render --skin 0 --face 20000 --hair 30000 --equip 1002000
python main.py --root_wz=Data --output_root=out character random --count 100 --seed 1 --weight SHIELD=0.05
python main.py --root_wz=Data character decode out/Character/c1_....png
python main.py --root_wz=Data --output_root=out generate_scene --config '{"num_samples": 10, "mob": {"count": 3}}'
python main.py --root_wz=Data --output_root=out generate_scene --config @scene.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from api import WzApi, node_to_json
from character.codec import decode
from character.equip import SKIN_COUNT, EquipType
from items import ItemCategory


def _item_category(text: str) -> ItemCategory:
    for category in ItemCategory:
        if text.lower() in (category.value.lower(), category.name.lower()):
            return category
    raise argparse.ArgumentTypeError(f"invalid category {text!r}; choose from {[c.value for c in ItemCategory]}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract mobs, maps, items, and characters from MapleStory data.")
    parser.add_argument("--root_wz", required=True, type=Path, help="Client Data directory.")
    parser.add_argument("--output_root", type=Path, default=None, help="Destination for dump commands.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v for info, -vv for debug logs.")
    sub = parser.add_subparsers(dest="kind", required=True)

    for kind in ("mob", "map", "item"):
        kind_parser = sub.add_parser(kind, help=f"{kind.capitalize()} commands.")
        actions = kind_parser.add_subparsers(dest="action", required=True)
        list_parser = actions.add_parser("list", help=f"Print (id, name) for every {kind}.")
        list_parser.add_argument("--json", action="store_true", help="Emit a JSON array instead of tuples.")
        dump_parser = actions.add_parser("dump", help=f"Export {kind}s under <output_root>/{kind.capitalize()}/.")
        dump_parser.add_argument("--ids", nargs="+", type=int, default=None, help="Ids to export (default: all).")
        if kind == "item":
            for action_parser in (list_parser, dump_parser):
                action_parser.add_argument(
                    "--category",
                    action="extend",
                    nargs="+",
                    type=_item_category,
                    metavar="{" + ",".join(c.value for c in ItemCategory) + "}",
                    help="Only these categories (case-insensitive).",
                )
        if kind == "map":
            dump_parser.add_argument(
                "--max_pixels", type=int, default=None, help="Downscale rendered maps above this pixel count."
            )
            dump_parser.add_argument(
                "--include_quest_items", action="store_true", help="Also draw quest-gated objects and backgrounds."
            )
            dump_parser.add_argument("--no_portals", action="store_true", help="Do not draw portals.")
            dump_parser.add_argument(
                "--include_hidden_portals", action="store_true", help="Also draw portals shown only up close."
            )

    scene_parser = sub.add_parser(
        "generate_scene", help="Render labeled synthetic scenes into <output_root>/Scenes/<timestamp>/."
    )
    scene_parser.add_argument(
        "--config", required=True, help="Scene config as a JSON string, or @path/to/config.json (see scene/config.py)."
    )
    scene_parser.add_argument("--no_catalog_cache", action="store_true", help="Always rebuild the character index.")

    char_parser = sub.add_parser("character", help="Character sprite commands.")
    char_parser.add_argument(
        "--catalog_cache",
        type=Path,
        default=None,
        help="Item index cache (default: <output_root>/.cache/character_catalog.json when --output_root is set).",
    )
    char_parser.add_argument("--no_catalog_cache", action="store_true", help="Always rebuild the item index.")
    char_actions = char_parser.add_subparsers(dest="action", required=True)
    slots_parser = char_actions.add_parser("list", help="Print the drawable ids of one slot.")
    slots_parser.add_argument("slot", type=str.upper, choices=[t.name for t in EquipType])
    render_parser = char_actions.add_parser("render", help="Render one character from slot ids.")
    render_parser.add_argument("--skin", type=int, required=True, help=f"Skin tone 0-{SKIN_COUNT - 1}.")
    render_parser.add_argument("--face", type=int, required=True)
    render_parser.add_argument("--hair", type=int, required=True)
    render_parser.add_argument(
        "--equip", nargs="+", type=int, default=[], help="Equipment ids; on conflicts, later ids win."
    )
    random_parser = char_actions.add_parser("random", help="Render random characters.")
    random_parser.add_argument("--count", type=int, default=1)
    random_parser.add_argument("--seed", type=int, default=None)
    random_parser.add_argument(
        "--weight", nargs="+", default=[], metavar="SLOT=P", help="Fill probability per optional slot, e.g. CAP=0.3."
    )
    random_parser.add_argument("--default_weight", type=float, default=0.5, help="Probability for unlisted slots.")
    random_parser.add_argument("--mixed_gender", action="store_true", help="Allow mixing male and female items.")
    decode_parser = char_actions.add_parser("decode", help="Print the slot ids encoded in character filenames.")
    decode_parser.add_argument("names", nargs="+", help="Filenames or paths such as c1_....png.")

    ls_parser = sub.add_parser("ls", help="Inspect a node, e.g. String/Mob.img/2.")
    ls_parser.add_argument("path", nargs="?", default="", help="Slash-separated node path.")
    ls_parser.add_argument("--json", action="store_true", help="Dump the subtree values as JSON.")
    return parser


def _print_list(items: list[tuple[int, str | None]], as_json: bool) -> None:
    if as_json:
        print(json.dumps([{"id": i, "name": n} for i, n in items], indent=2, ensure_ascii=False))
        return
    for item in items:
        print(item)


def _ls(api: WzApi, path: str, as_json: bool) -> int:
    node = api.node(path) if path else api.wz.root
    if node is None:
        print(f"not found: {path}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(node_to_json(node), indent=2, ensure_ascii=False))
        return 0
    if len(node) == 0:
        print(f"{node.name} = {node.value!r}")
    for child in node:
        suffix = "/" if len(child.children) > 0 or child.is_image else ""
        value = "" if child.value is None else f" = {child.value!r}"
        print(f"{child.name}{suffix}{value}")
    return 0


def _parse_weights(items: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Expected SLOT=P, got {item!r}")
        weights[name] = float(value)
    return weights


def _character(api: WzApi, args: argparse.Namespace) -> int:
    char = api.character
    if args.action == "list":
        for item_id in char.ids(EquipType[args.slot]):
            print(item_id)
        return 0
    if args.action == "render":
        assignments = [(EquipType.SKIN, args.skin), (EquipType.FACE, args.face), (EquipType.HAIR, args.hair)]
        assignments += [(None, item_id) for item_id in args.equip]
        written = char.dump([char.build(assignments)])
    else:
        char.default_weight = args.default_weight
        written = char.dump_random(
            args.count, args.seed, _parse_weights(args.weight), consistent_gender=not args.mixed_gender
        )
    for path in written:
        print(path)
    return 0 if written else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = _build_parser().parse_args(argv)
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    if args.kind == "character" and args.action == "decode":
        try:
            for name in args.names:
                print(json.dumps({"name": Path(name).name, **decode(name).to_dict()}))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        return 0
    if not args.root_wz.is_dir():
        print(f"--root_wz is not a directory: {args.root_wz}", file=sys.stderr)
        return 2
    if getattr(args, "action", None) in ("dump", "render", "random") and args.output_root is None:
        print(f"--output_root is required for {args.action}", file=sys.stderr)
        return 2
    if args.kind == "generate_scene" and args.output_root is None:
        print("--output_root is required for generate_scene", file=sys.stderr)
        return 2

    cache = None
    if args.kind in ("character", "generate_scene") and not args.no_catalog_cache:
        cache = getattr(args, "catalog_cache", None)
        if cache is None and args.output_root is not None:
            cache = args.output_root / ".cache" / "character_catalog.json"

    with WzApi(args.root_wz, args.output_root, character_catalog_cache=cache) as api:
        if args.kind == "ls":
            return _ls(api, args.path, args.json)
        if args.kind == "generate_scene":
            try:
                text = args.config
                if text.startswith("@"):
                    text = Path(text[1:]).read_text(encoding="utf-8")
                print(api.generate_scene(text))
            except (ValueError, OSError) as exc:
                print(exc, file=sys.stderr)
                return 2
            return 0
        if args.kind == "character":
            try:
                return _character(api, args)
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 2
        if args.kind == "item":
            if args.action == "list":
                _print_list(api.item.list(args.category), args.json)
                return 0
            written = api.item.dump(args.ids, args.category)
            for path in written:
                print(path)
            wanted = set(args.category or ItemCategory)
            requested = {i for i in (args.ids or api.item.ids()) if api.item.category(i) in wanted}
            return 0 if len(written) == len(requested) else 1
        sub_api = api.mob if args.kind == "mob" else api.map
        if args.action == "list":
            _print_list(sub_api.list(), args.json)
            return 0
        if args.kind == "map":
            if args.max_pixels is not None:
                api.map.renderer.max_pixels = args.max_pixels
            api.map.renderer.include_quest_items = args.include_quest_items
            api.map.renderer.include_portals = not args.no_portals
            api.map.renderer.include_hidden_portals = args.include_hidden_portals
        written = sub_api.dump(args.ids)
        for path in written:
            print(path)
        requested = len(set(args.ids)) if args.ids is not None else len(sub_api.ids())
        return 0 if len(written) == requested else 1


if __name__ == "__main__":
    sys.exit(main())
