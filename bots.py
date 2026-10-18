"""
bots.py  --  simulated opponents used in the experiments.

Real people are slow and inconsistent, so we test with bots that follow a
known pattern. That makes the results repeatable and lets us see exactly
which pattern each algorithm can (or cannot) learn.

Every bot has one method:
    next_move(ai_last)   ai_last = the AI's move in the previous round (or None)
"""

import random
from game import MOVES, BEATS


class RandomBot:
    """Completely random. No AI can beat this (expected win rate = 33%)."""
    name = "Random"

    def next_move(self, ai_last=None):
        return random.choice(MOVES)


class BiasedBot:
    """Plays Rock 50% of the time, Paper and Scissors 25% each."""
    name = "Biased (50% Rock)"

    def next_move(self, ai_last=None):
        return random.choices(MOVES, weights=[0.5, 0.25, 0.25])[0]


class CyclicBot:
    """Plays R, P, S, R, P, S, ... forever."""
    name = "Cyclic (R,P,S)"

    def __init__(self):
        self.i = -1

    def next_move(self, ai_last=None):
        self.i += 1
        return MOVES[self.i % 3]


class SwitchBot:
    """Never repeats its own previous move; picks randomly between the other two.
    Frequency counting cannot see this pattern, but a Markov model can."""
    name = "Switcher (never repeats)"

    def __init__(self):
        self.last = None

    def next_move(self, ai_last=None):
        if self.last is None:
            move = random.choice(MOVES)
        else:
            move = random.choice([m for m in MOVES if m != self.last])
        self.last = move
        return move


class ReactiveBot:
    """Plays the move that would have beaten the AI's previous move.
    This is a HARD opponent for our two predictors, because its move depends
    on what the AI did, not on its own history. Good for the 'limitations'
    section of the report."""
    name = "Reactive (beats AI's last)"

    def next_move(self, ai_last=None):
        if ai_last is None:
            return random.choice(MOVES)
        return BEATS[ai_last]


ALL_BOTS = [RandomBot, BiasedBot, CyclicBot, SwitchBot, ReactiveBot]
