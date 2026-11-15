"""A reshuffling deck so that every card is drawn once per cycle."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Generic, TypeVar

T = TypeVar("T")


class Deck(Generic[T]):
    """Draws cards without replacement; when empty, the full set is reshuffled and drawing continues.

    Over `k * len(cards)` draws every card appears exactly `k` times, which spreads samples evenly
    over the whole set instead of relying on independent random picks.

    Raises
    ------
    ValueError
        If `cards` is empty.
    """

    def __init__(self, cards: Sequence[T], rng: random.Random) -> None:
        if not cards:
            raise ValueError("Cannot build a deck without cards")
        self._cards = list(cards)
        self._rng = rng
        self._stack: list[T] = []
        self.cycles = 0
        """Number of times the deck has been (re)shuffled."""

    def __len__(self) -> int:
        return len(self._cards)

    @property
    def remaining(self) -> int:
        """Cards left before the next reshuffle."""
        return len(self._stack)

    def draw(self) -> T:
        """Pop the next card, reshuffling the full set first if the stack is empty."""
        if not self._stack:
            self._stack = list(self._cards)
            self._rng.shuffle(self._stack)
            self.cycles += 1
        return self._stack.pop()
