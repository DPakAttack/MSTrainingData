"""Pure helpers for combining slot assignments and sampling random characters."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence

from character.catalog import ItemInfo
from character.codec import CharacterSpec
from character.equip import OPTIONAL_SLOTS, REQUIRED_SLOTS, EquipType

InfoLookup = Callable[[EquipType, int], "ItemInfo | None"]


def resolve_assignments(assignments: Iterable[tuple[EquipType, int]], lookup: InfoLookup) -> CharacterSpec:
    """Apply assignments in order; each replaces its slot and evicts earlier items it conflicts with.

    Two items conflict when their `islot` codes overlap (for example an overall `MaPn` and a top
    `Ma`, or a two-handed weapon `WpSi` and a shield `Si`). The latest assignment wins. Skin, face,
    and hair are never evicted and never evict equipment (some caps declare `HrCp`); they only
    replace an earlier assignment of their own slot.

    Raises
    ------
    ValueError
        If an id is not a drawable item of its slot.
    """
    chosen: dict[EquipType, ItemInfo] = {}
    for slot, item_id in assignments:
        info = lookup(slot, item_id)
        if info is None:
            raise ValueError(f"{item_id} is not a drawable {slot.value}")
        codes = set(info.islot)
        for other_slot, other in list(chosen.items()):
            optional_pair = slot not in REQUIRED_SLOTS and other_slot not in REQUIRED_SLOTS
            if other_slot == slot or (optional_pair and codes & set(other.islot)):
                del chosen[other_slot]
        chosen[slot] = info
    return CharacterSpec.from_mapping({slot: info.item_id for slot, info in chosen.items()})


def sample_styled(rng: random.Random, ids: Sequence[int], style: Callable[[int], int]) -> int:
    """Pick a style uniformly, then one of that style's ids (its colors) uniformly.

    Raises
    ------
    ValueError
        If `ids` is empty.
    """
    if not ids:
        raise ValueError("Cannot sample from an empty pool")
    groups: dict[int, list[int]] = {}
    for item_id in sorted(ids):
        groups.setdefault(style(item_id), []).append(item_id)
    styles = sorted(groups)
    return rng.choice(groups[rng.choice(styles)])


def normalize_weights(weight_map: Mapping[EquipType | str, float] | None, default: float) -> dict[EquipType, float]:
    """Validate `weight_map` and fill every optional slot (unlisted slots get `default`).

    Keys may be `EquipType` members or their names/values (`"CAP"`, `"cap"`).

    Raises
    ------
    ValueError
        For unknown or required slots, or probabilities outside `[0, 1]`.
    """
    weights = dict.fromkeys(OPTIONAL_SLOTS, default)
    for key, value in (weight_map or {}).items():
        slot = key if isinstance(key, EquipType) else _parse_slot(key)
        if slot in REQUIRED_SLOTS:
            raise ValueError(f"{slot.value} is always present and cannot be weighted")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Weight for {slot.value} must be in [0, 1], got {value}")
        weights[slot] = float(value)
    if not 0.0 <= default <= 1.0:
        raise ValueError(f"Default weight must be in [0, 1], got {default}")
    return weights


def _parse_slot(name: str) -> EquipType:
    try:
        return EquipType[name.upper()]
    except KeyError:
        try:
            return EquipType(name.lower())
        except ValueError:
            raise ValueError(f"Unknown slot {name!r}; expected one of {[t.name for t in EquipType]}") from None


def sample_optional(
    rng: random.Random, pools: Mapping[EquipType, Sequence[int]], weights: Mapping[EquipType, float]
) -> list[tuple[EquipType, int]]:
    """Choose optional slot assignments in a shuffled slot order.

    Each slot is visited once, in an order shuffled by `rng`, and filled with probability
    `weights[slot]` using an item drawn uniformly from its pool. The random draws do not depend
    on the weights, so changing one slot's weight does not reshuffle the other slots.
    """
    order = sorted(OPTIONAL_SLOTS, key=lambda s: s.value)
    rng.shuffle(order)
    out: list[tuple[EquipType, int]] = []
    for slot in order:
        roll = rng.random()
        pool = pools.get(slot) or ()
        pick = rng.choice(pool) if pool else None
        if pick is not None and roll < weights.get(slot, 0.0):
            out.append((slot, pick))
    return out
