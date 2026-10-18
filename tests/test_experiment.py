from adaptive_rps.experiment import run_experiments, summarize


def test_experiment_shape_and_reproducibility():
    first = run_experiments(rounds=10, seeds=1, switch_round=5)
    second = run_experiments(rounds=10, seeds=1, switch_round=5)
    assert len(first) == 10 * 1 * 4 * 3
    assert first[["agent_move", "opponent_move", "outcome"]].equals(
        second[["agent_move", "opponent_move", "outcome"]]
    )
    assert not summarize(first).empty
