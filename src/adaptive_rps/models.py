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

    def get_next_move_probabilities(self) -> dict[Move, float]:
        """Return uniform probability distribution."""
        return {m: 1.0 / len(MOVES) for m in MOVES}


class FrequencyAgent:
    """Counters the opponent's most frequent observed move."""
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

    def get_next_move_probabilities(self) -> dict[Move, float]:
        """Return empirical frequency distribution."""
        if not self.counts:
            return {m: 1.0 / len(MOVES) for m in MOVES}
        total = sum(self.counts.values())
        return {m: self.counts[m] / total for m in MOVES}


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

    def get_transition_probabilities(self) -> dict[Move, dict[Move, float]]:
        """Return row-normalized transition probabilities P(next | current)."""
        probs: dict[Move, dict[Move, float]] = {}
        for m1 in MOVES:
            row = self.transitions.get(m1, Counter())
            row_sum = sum(row.values())
            if row_sum > 0:
                probs[m1] = {m2: row[m2] / row_sum for m2 in MOVES}
            else:
                probs[m1] = {m2: 1.0 / len(MOVES) for m2 in MOVES}
        return probs

    def get_next_move_probabilities(self) -> dict[Move, float]:
        """Return the probability distribution over the opponent's next move."""
        counts = self.transitions.get(self.previous_move, Counter()) if self.previous_move is not None else Counter()
        if not counts:
            counts = self.global_counts
        if not counts:
            return {m: 1.0 / len(MOVES) for m in MOVES}
        total = sum(counts.values())
        return {m: counts[m] / total for m in MOVES}


class DecayMarkovAgent:
    """First-order Markov model with exponential decay factor gamma in (0, 1) for concept drift."""
    name = "Decayed Markov model"

    def __init__(self, seed: int = 0, gamma: float = 0.90) -> None:
        if not (0.0 < gamma < 1.0):
            raise ValueError("gamma must be between 0 and 1")
        self.rng = random.Random(seed)
        self.gamma = gamma
        self.transitions: dict[Move, dict[Move, float]] = {
            m1: {m2: 0.0 for m2 in MOVES} for m1 in MOVES
        }
        self.previous_move: Move | None = None
        self.global_counts: dict[Move, float] = {m: 0.0 for m in MOVES}

    def choose(self, history: Sequence[Move]) -> tuple[Move, Move | None]:
        counts = self.transitions.get(self.previous_move, {}) if self.previous_move is not None else {}
        total = sum(counts.values()) if counts else 0.0
        if total <= 1e-6:
            counts = self.global_counts
            total = sum(counts.values())
        if total <= 1e-6:
            return self.rng.choice(MOVES), None

        max_val = max(counts.values())
        candidates = [m for m in MOVES if abs(counts[m] - max_val) < 1e-9]
        prediction = self.rng.choice(candidates)
        return counter_move(prediction), prediction

    def observe(self, move: Move) -> None:
        # Exponentially decay past transition counts and global counts
        for m1 in MOVES:
            for m2 in MOVES:
                self.transitions[m1][m2] *= self.gamma
            self.global_counts[m1] *= self.gamma

        # Increment observed transition and global count
        if self.previous_move is not None:
            self.transitions[self.previous_move][move] += 1.0
        self.global_counts[move] += 1.0
        self.previous_move = move

    def get_transition_probabilities(self) -> dict[Move, dict[Move, float]]:
        """Return row-normalized transition probabilities P(next | current)."""
        probs: dict[Move, dict[Move, float]] = {}
        for m1 in MOVES:
            row_sum = sum(self.transitions[m1].values())
            if row_sum > 1e-6:
                probs[m1] = {m2: self.transitions[m1][m2] / row_sum for m2 in MOVES}
            else:
                probs[m1] = {m2: 1.0 / len(MOVES) for m2 in MOVES}
        return probs

    def get_next_move_probabilities(self) -> dict[Move, float]:
        """Return the probability distribution over the opponent's next move."""
        if self.previous_move is not None:
            row = self.transitions[self.previous_move]
            total = sum(row.values())
            if total > 1e-6:
                return {m: row[m] / total for m in MOVES}
        total_global = sum(self.global_counts.values())
        if total_global > 1e-6:
            return {m: self.global_counts[m] / total_global for m in MOVES}
        return {m: 1.0 / len(MOVES) for m in MOVES}


class Markov2Agent:
    """Second-order Markov model: conditions prediction on the opponent's last two moves."""
    name = "Second-order Markov model"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.transitions: dict[tuple[Move, Move], Counter[Move]] = defaultdict(Counter)
        self.transitions_order1: dict[Move, Counter[Move]] = defaultdict(Counter)
        self.global_counts: Counter[Move] = Counter()
        self.history: list[Move] = []

    def choose(self, history: Sequence[Move]) -> tuple[Move, Move | None]:
        counts: Counter[Move] | None = None
        # Try second-order transition
        if len(self.history) >= 2:
            key2 = (self.history[-2], self.history[-1])
            if self.transitions.get(key2):
                counts = self.transitions[key2]

        # Fallback to first-order transition
        if not counts and len(self.history) >= 1:
            key1 = self.history[-1]
            if self.transitions_order1.get(key1):
                counts = self.transitions_order1[key1]

        # Fallback to global counts
        if not counts and self.global_counts:
            counts = self.global_counts

        if not counts:
            return self.rng.choice(MOVES), None

        max_count = max(counts.values())
        candidates = [m for m in MOVES if counts[m] == max_count]
        prediction = self.rng.choice(candidates)
        return counter_move(prediction), prediction

    def observe(self, move: Move) -> None:
        if len(self.history) >= 2:
            self.transitions[(self.history[-2], self.history[-1])][move] += 1
        if len(self.history) >= 1:
            self.transitions_order1[self.history[-1]][move] += 1
        self.global_counts[move] += 1
        self.history.append(move)

    def get_next_move_probabilities(self) -> dict[Move, float]:
        """Return the probability distribution over the opponent's next move."""
        counts = None
        if len(self.history) >= 2:
            key2 = (self.history[-2], self.history[-1])
            if self.transitions.get(key2):
                counts = self.transitions[key2]
        if not counts and len(self.history) >= 1:
            key1 = self.history[-1]
            if self.transitions_order1.get(key1):
                counts = self.transitions_order1[key1]
        if not counts and self.global_counts:
            counts = self.global_counts
        if not counts:
            return {m: 1.0 / len(MOVES) for m in MOVES}
        total = sum(counts.values())
        return {m: counts[m] / total for m in MOVES}
