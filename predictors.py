"""
predictors.py  --  the "AI" part of the project.

Every predictor has the same three methods:
    predict()            -> the opponent's most likely next move (or None if no data yet)
    update(opp_move)     -> learn from the move the opponent just played
    choose_move()        -> the AI's own move: the counter to the prediction
                            (a random move if there is nothing to predict yet)

Three predictors:
    RandomPredictor     baseline, always plays randomly (a "no learning" control)
    FrequencyPredictor  Algorithm A: predicts the opponent's most common move
    MarkovPredictor     Algorithm B: predicts from the opponent's previous move
"""

import random
from game import MOVES, BEATS


def _argmax_random_tie(counts):
    """Return the key with the largest count. Ties are broken randomly, and
    if every count is 0 (no data) we return None."""
    best = max(counts.values())
    if best == 0:
        return None
    return random.choice([m for m, c in counts.items() if c == best])


class Predictor:
    name = "base"

    def predict(self):
        raise NotImplementedError

    def update(self, opp_move):
        raise NotImplementedError

    def choose_move(self):
        prediction = self.predict()
        if prediction is None:              # no data yet -> play randomly
            return random.choice(MOVES)
        return BEATS[prediction]            # play the move that beats the prediction


class RandomPredictor(Predictor):
    """Baseline: never learns. Should win about 1/3 of the time against anyone."""
    name = "Random (baseline)"

    def predict(self):
        return None

    def update(self, opp_move):
        pass


class FrequencyPredictor(Predictor):
    """Algorithm A: count how often the opponent plays each move."""
    name = "Frequency"

    def __init__(self):
        self.counts = {m: 0 for m in MOVES}

    def predict(self):
        return _argmax_random_tie(self.counts)

    def update(self, opp_move):
        self.counts[opp_move] += 1


class MarkovPredictor(Predictor):
    """Algorithm B: first-order Markov model.

    Remembers what the opponent played AFTER each previous move, e.g.
    transitions["R"] = {"R": 2, "P": 7, "S": 1} means: after Rock, this
    opponent played Paper 7 times, so predict Paper next time they play Rock.
    """
    name = "Markov"

    def __init__(self):
        self.transitions = {a: {b: 0 for b in MOVES} for a in MOVES}
        self.last = None  # the opponent's previous move

    def predict(self):
        if self.last is None:
            return None
        return _argmax_random_tie(self.transitions[self.last])

    def update(self, opp_move):
        if self.last is not None:
            self.transitions[self.last][opp_move] += 1
        self.last = opp_move
