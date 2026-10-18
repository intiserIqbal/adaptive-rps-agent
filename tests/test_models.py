from adaptive_rps.models import FrequencyAgent, MarkovAgent


def test_frequency_agent_learns_fixed_move():
    agent = FrequencyAgent(seed=1)
    for _ in range(20):
        agent.observe("rock")
    move, prediction = agent.choose(["rock"] * 20)
    assert prediction == "rock"
    assert move == "paper"


def test_markov_agent_learns_transition():
    agent = MarkovAgent(seed=1)
    # Repeated rock -> paper transitions teach the model to predict paper after rock.
    for move in ["rock", "paper", "rock", "paper", "rock", "paper"]:
        agent.observe(move)
    agent.previous_move = "rock"
    move, prediction = agent.choose([])
    assert prediction == "paper"
    assert move == "scissors"
