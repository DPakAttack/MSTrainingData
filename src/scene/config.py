"""JSON configuration for `generate_scene`.

Example
-------
{
  "resolution": "1366x768",              # or [1366, 768]; default 1366x768
  "num_samples": 100,                    # required, >= 1
  "seed": 7,                             # optional; null picks a random seed (recorded in config.json)
  "character": {"min_count": 1, "max_count": 3, "equip_weights": {"CAP": 0.4, "SHIELD": 0.05}},
  "mob": {"count": 4},
  "item": {"min_count": 0, "max_count": 6, "meso_fraction": 0.5},
  "flip_probability": 0.5,               # chance each character / mob / item is mirrored
  "maps": null                           # optional list of map ids to draw from
}

Each of `character`, `mob`, `item` takes either `count`, or `min_count` and `max_count` (the per-scene
count is then uniform over the inclusive range). An omitted category is never drawn.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from character.build import normalize_weights
from character.equip import EquipType

_RESOLUTION_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")
DEFAULT_RESOLUTION = (1366, 768)


@dataclass(frozen=True)
class CountSpec:
    """How many instances of a category appear per scene (inclusive range)."""

    min_count: int = 0
    max_count: int = 0

    def sample(self, rng: random.Random) -> int:
        """A uniformly drawn count in `[min_count, max_count]`."""
        return rng.randint(self.min_count, self.max_count)

    def to_json(self) -> dict[str, int]:
        if self.min_count == self.max_count:
            return {"count": self.min_count}
        return {"min_count": self.min_count, "max_count": self.max_count}


@dataclass(frozen=True)
class SceneConfig:
    """Validated scene-generation settings (see the module docstring for the JSON form)."""

    num_samples: int
    resolution: tuple[int, int] = DEFAULT_RESOLUTION
    seed: int | None = None
    characters: CountSpec = field(default_factory=CountSpec)
    equip_weights: dict[str, float] | None = None
    mobs: CountSpec = field(default_factory=CountSpec)
    items: CountSpec = field(default_factory=CountSpec)
    meso_fraction: float = 0.5
    flip_probability: float = 0.5
    maps: tuple[int, ...] | None = None

    def with_seed(self, seed: int) -> SceneConfig:
        """Copy with `seed` fixed."""
        return SceneConfig(**{**self.__dict__, "seed": seed})

    def to_json(self) -> dict[str, Any]:
        """The config in its JSON form (round-trips through `parse_config`)."""
        character: dict[str, Any] = self.characters.to_json()
        if self.equip_weights is not None:
            character["equip_weights"] = dict(self.equip_weights)
        return {
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "num_samples": self.num_samples,
            "seed": self.seed,
            "character": character,
            "mob": self.mobs.to_json(),
            "item": {**self.items.to_json(), "meso_fraction": self.meso_fraction},
            "flip_probability": self.flip_probability,
            "maps": list(self.maps) if self.maps is not None else None,
        }


def _int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


def _reject_unknown(section: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys {sorted(unknown)}; allowed: {sorted(allowed)}")


def parse_count(section: Any, name: str, extra_keys: frozenset[str] = frozenset()) -> CountSpec:
    """Parse `{"count": n}` or `{"min_count": a, "max_count": b}`; `None` means zero.

    Raises
    ------
    ValueError
        If both or neither forms are given, values are negative, or `min_count > max_count`.
    """
    if section is None:
        return CountSpec()
    if not isinstance(section, Mapping):
        raise ValueError(f"{name} must be an object, got {section!r}")
    _reject_unknown(section, {"count", "min_count", "max_count"} | extra_keys, name)
    has_count = "count" in section
    has_range = "min_count" in section or "max_count" in section
    if has_count == has_range:
        raise ValueError(f"{name} needs either 'count' or both 'min_count' and 'max_count'")
    if has_count:
        n = _int(section["count"], f"{name}.count")
        return CountSpec(n, n)
    if "min_count" not in section or "max_count" not in section:
        raise ValueError(f"{name} needs both 'min_count' and 'max_count'")
    lo = _int(section["min_count"], f"{name}.min_count")
    hi = _int(section["max_count"], f"{name}.max_count")
    if lo > hi:
        raise ValueError(f"{name}.min_count ({lo}) is greater than max_count ({hi})")
    return CountSpec(lo, hi)


def parse_resolution(value: Any) -> tuple[int, int]:
    """Parse `"WxH"` or `[W, H]`."""
    if value is None:
        return DEFAULT_RESOLUTION
    if isinstance(value, str) and (m := _RESOLUTION_RE.match(value)):
        size = (int(m.group(1)), int(m.group(2)))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        size = (_int(value[0], "resolution[0]", 1), _int(value[1], "resolution[1]", 1))
    else:
        raise ValueError(f"resolution must be 'WxH' or [W, H], got {value!r}")
    if min(size) < 1:
        raise ValueError(f"resolution must be positive, got {value!r}")
    return size


def parse_config(config: str | Mapping[str, Any] | SceneConfig) -> SceneConfig:
    """Validate a JSON string or mapping into a `SceneConfig`.

    Raises
    ------
    ValueError
        With a message naming the offending field.
    """
    if isinstance(config, SceneConfig):
        return config
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Scene config is not valid JSON: {exc}") from None
    if not isinstance(config, Mapping):
        raise ValueError("Scene config must be a JSON object")
    _reject_unknown(
        config,
        {"resolution", "num_samples", "seed", "character", "mob", "item", "flip_probability", "maps"},
        "config",
    )
    if "num_samples" not in config:
        raise ValueError("num_samples is required")
    character = config.get("character")
    weights = None
    if isinstance(character, Mapping) and character.get("equip_weights") is not None:
        raw = character["equip_weights"]
        if not isinstance(raw, Mapping):
            raise ValueError("character.equip_weights must be an object of slot -> weight")
        normalize_weights(raw, 0.5)
        weights = {(k.name if isinstance(k, EquipType) else str(k)): float(v) for k, v in raw.items()}
    item = config.get("item")
    meso_fraction = 0.5
    if isinstance(item, Mapping) and "meso_fraction" in item:
        meso_fraction = _probability(item["meso_fraction"], "item.meso_fraction")
    seed = config.get("seed")
    if seed is not None:
        seed = _int(seed, "seed")
    maps = config.get("maps")
    if maps is not None:
        if not isinstance(maps, (list, tuple)) or not maps:
            raise ValueError("maps must be a non-empty list of map ids or null")
        maps = tuple(_int(m, "maps[]") for m in maps)
    return SceneConfig(
        num_samples=_int(config["num_samples"], "num_samples", 1),
        resolution=parse_resolution(config.get("resolution")),
        seed=seed,
        characters=parse_count(character, "character", frozenset({"equip_weights"})),
        equip_weights=weights,
        mobs=parse_count(config.get("mob"), "mob"),
        items=parse_count(item, "item", frozenset({"meso_fraction"})),
        meso_fraction=meso_fraction,
        flip_probability=_probability(config.get("flip_probability", 0.5), "flip_probability"),
        maps=maps,
    )
