from adaptive_rps.experiment import run_experiments, summarize


def test_experiment_shape_and_reproducibility():
    first = run_experiments(rounds=10, seeds=1, switch_round=5)
    second = run_experiments(rounds=10, seeds=1, switch_round=5)
    # 10 rounds * 1 seed * 4 opponents * 5 agent models
    assert len(first) == 10 * 1 * 4 * 5
    assert first[["agent_move", "opponent_move", "outcome"]].equals(
        second[["agent_move", "opponent_move", "outcome"]]
    )
    assert not summarize(first).empty


def test_experiment_selected_agents():
    results = run_experiments(
        rounds=5, seeds=1, switch_round=2,
        selected_agents=["Frequency model", "Decayed Markov model (γ=0.90)"]
    )
    # 5 rounds * 1 seed * 4 opponents * 2 agents
    assert len(results) == 5 * 1 * 4 * 2
