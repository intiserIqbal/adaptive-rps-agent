"""Controlled opponents used for reproducible experiments."""
from __future__ import annotations

import random
from collections.abc import Sequence

from adaptive_rps.game import MOVES, Move


class RandomOpponent:
    """Chooses each move uniformly at random."""
    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def play(self, history: Sequence[Move]) -> Move:
        return self.rng.choice(MOVES)


class BiasedOpponent:
    """Uses a fixed, configurable distribution; defaults to favoring rock."""
    name = "Biased (rock-heavy)"

    def __init__(self, seed: int = 0, weights: tuple[float, float, float] = (0.7, 0.2, 0.1)) -> None:
        if len(weights) != 3 or any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("weights must be three non-negative values with a positive sum")
        self.rng = random.Random(seed)
        self.weights = weights

    def play(self, history: Sequence[Move]) -> Move:
        return self.rng.choices(MOVES, weights=self.weights, k=1)[0]


class CycleOpponent:
    """Repeats rock, paper, scissors in that order."""
    name = "Cyclic pattern"

    def play(self, history: Sequence[Move]) -> Move:
        return MOVES[len(history) % len(MOVES)]


class SwitchingOpponent:
    """Starts rock-heavy, then switches to scissors-heavy halfway through."""
    name = "Strategy switch"

    def __init__(self, seed: int = 0, switch_round: int = 100) -> None:
        self.rng = random.Random(seed)
        self.switch_round = switch_round

    def play(self, history: Sequence[Move]) -> Move:
        weights = (0.7, 0.2, 0.1) if len(history) < self.switch_round else (0.1, 0.2, 0.7)
        return self.rng.choices(MOVES, weights=weights, k=1)[0]
