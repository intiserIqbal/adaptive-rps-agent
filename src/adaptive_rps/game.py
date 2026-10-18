"""Core Rock-Paper-Scissors rules."""
from __future__ import annotations

from typing import Literal

Move = Literal["rock", "paper", "scissors"]
MOVES: tuple[Move, ...] = ("rock", "paper", "scissors")
BEATS: dict[Move, Move] = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


def outcome(agent_move: Move, opponent_move: Move) -> str:
    """Return 'win', 'draw', or 'loss' from the agent's perspective."""
    if agent_move == opponent_move:
        return "draw"
    return "win" if BEATS[agent_move] == opponent_move else "loss"


def counter_move(predicted_move: Move) -> Move:
    """Return the move that beats the predicted opponent move."""
    return next(move for move, beaten in BEATS.items() if beaten == predicted_move)
