from adaptive_rps.models import (
    DecayMarkovAgent,
    FrequencyAgent,
    Markov2Agent,
    MarkovAgent,
    RandomAgent,
)


def test_frequency_agent_learns_fixed_move():
    agent = FrequencyAgent(seed=1)
    for _ in range(20):
        agent.observe("rock")
    move, prediction = agent.choose(["rock"] * 20)
    assert prediction == "rock"
    assert move == "paper"
    probs = agent.get_next_move_probabilities()
    assert probs["rock"] == 1.0


def test_markov_agent_learns_transition():
    agent = MarkovAgent(seed=1)
    # Repeated rock -> paper transitions teach the model to predict paper after rock.
    for move in ["rock", "paper", "rock", "paper", "rock", "paper"]:
        agent.observe(move)
    agent.previous_move = "rock"
    move, prediction = agent.choose([])
    assert prediction == "paper"
    assert move == "scissors"
    probs = agent.get_transition_probabilities()
    assert probs["rock"]["paper"] > 0.9


def test_decay_markov_agent_adapts_to_shift():
    # Standard Markov without decay struggles to quickly shift from 100 observations.
    # Decayed Markov with gamma=0.8 decays past counts rapidly.
    agent = DecayMarkovAgent(seed=1, gamma=0.85)
    # Phase 1: 30 observations of rock -> paper
    for _ in range(30):
        agent.observe("rock")
        agent.observe("paper")
    # Phase 2: 15 observations of rock -> scissors
    for _ in range(15):
        agent.observe("rock")
        agent.observe("scissors")
    agent.previous_move = "rock"
    move, prediction = agent.choose([])
    # Scissors should now dominate due to decay of earlier paper transitions
    assert prediction == "scissors"
    assert move == "rock"


def test_markov2_agent_learns_sequence():
    agent = Markov2Agent(seed=1)
    # Sequence: rock -> paper -> scissors -> rock -> paper -> scissors
    sequence = ["rock", "paper", "scissors", "rock", "paper", "scissors", "rock", "paper", "scissors"]
    for m in sequence:
        agent.observe(m)
    # The last two moves in history are paper -> scissors. Next was rock in sequence.
    # Let's set history explicitly to [..., "rock", "paper"]
    agent.history = ["rock", "paper"]
    move, prediction = agent.choose([])
    assert prediction == "scissors"
    assert move == "rock"


def test_random_agent_probabilities():
    agent = RandomAgent(seed=1)
    probs = agent.get_next_move_probabilities()
    assert len(probs) == 3
    for p in probs.values():
        assert abs(p - 1.0 / 3.0) < 1e-6
