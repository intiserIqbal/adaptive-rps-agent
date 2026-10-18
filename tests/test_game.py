from adaptive_rps.game import counter_move, outcome


def test_all_outcomes():
    assert outcome("rock", "scissors") == "win"
    assert outcome("paper", "rock") == "win"
    assert outcome("scissors", "paper") == "win"
    assert outcome("rock", "rock") == "draw"
    assert outcome("rock", "paper") == "loss"


def test_counter_move():
    assert counter_move("rock") == "paper"
    assert counter_move("paper") == "scissors"
    assert counter_move("scissors") == "rock"
