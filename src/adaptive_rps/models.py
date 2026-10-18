"""Opponent models and agents. Each prediction uses only prior observations."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence

from adaptive_rps.game import MOVES, Move, counter_move


class RandomAgent:
    """Non-learning baseline that chooses uniformly at random."""
    name = "Random baseline"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def choose(self, history: Sequence[Move]) -> tuple[Move, Move | None]:
        return self.rng.choice(MOVES), None

    def observe(self, move: Move) -> None:
        pass


class FrequencyAgent:
    """Counters the opponent's most frequent observed move, with Laplace smoothing."""
    name = "Frequency model"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.counts: Counter[Move] = Counter()

    def choose(self, history: Sequence[Move]) -> tuple[Move, Move | None]:
        if not self.counts:
            return self.rng.choice(MOVES), None
        max_count = max(self.counts.values())
        candidates = [move for move in MOVES if self.counts[move] == max_count]
        prediction = self.rng.choice(candidates)
        return counter_move(prediction), prediction

    def observe(self, move: Move) -> None:
        self.counts[move] += 1


class MarkovAgent:
    """Counters the most common next move after the opponent's previous move."""
    name = "First-order Markov model"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.transitions: dict[Move, Counter[Move]] = defaultdict(Counter)
        self.previous_move: Move | None = None
        self.global_counts: Counter[Move] = Counter()

    def choose(self, history: Sequence[Move]) -> tuple[Move, Move | None]:
        counts = self.transitions.get(self.previous_move, Counter()) if self.previous_move is not None else Counter()
        if not counts:
            counts = self.global_counts
        if not counts:
            return self.rng.choice(MOVES), None
        max_count = max(counts.values())
        candidates = [move for move in MOVES if counts[move] == max_count]
        prediction = self.rng.choice(candidates)
        return counter_move(prediction), prediction

    def observe(self, move: Move) -> None:
        if self.previous_move is not None:
            self.transitions[self.previous_move][move] += 1
        self.global_counts[move] += 1
        self.previous_move = move
