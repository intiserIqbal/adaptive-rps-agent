"""
game.py  --  the rules of Rock-Paper-Scissors.

Everything else in the project imports from this file, so it is the
"environment" part of your report.
"""

MOVES = ["R", "P", "S"]  # Rock, Paper, Scissors

# BEATS[x] gives the move that BEATS x.
#   Paper beats Rock, Scissors beats Paper, Rock beats Scissors.
BEATS = {"R": "P", "P": "S", "S": "R"}

NAMES = {"R": "Rock", "P": "Paper", "S": "Scissors"}


def result(ai_move, opp_move):
    """Return the result from the AI's point of view.

    +1 = AI wins, 0 = draw, -1 = AI loses.
    """
    if ai_move == opp_move:
        return 0
    if BEATS[opp_move] == ai_move:  # the AI's move beats the opponent's move
        return 1
    return -1
