from pathlib import Path

import pytest

from adaptive_rps.experiment import (
    evaluate_adaptation,
    export_results,
    run_experiments,
    run_match,
    summarize,
    summarize_adaptation,
    validate_experiment_fairness,
)
from adaptive_rps.models import FrequencyAgent, RandomAgent
from adaptive_rps.opponents import BiasedOpponent, RandomOpponent


def test_experiment_shape_and_reproducibility():
    first = run_experiments(rounds=10, seeds=1, switch_round=5)
    second = run_experiments(rounds=10, seeds=1, switch_round=5)
    assert len(first) == 10 * 1 * 4 * 3
    assert first[["agent_move", "opponent_move", "outcome"]].equals(
        second[["agent_move", "opponent_move", "outcome"]]
    )
    assert not summarize(first).empty


def test_rolling_metrics_in_run_match():
    agent = FrequencyAgent(seed=42)
    opponent = BiasedOpponent(seed=42, weights=(1.0, 0.0, 0.0))  # Always throws rock
    df = run_match(agent, opponent, rounds=30, seed=1, opponent_name="RockOnly", agent_name="Freq", rolling_window=10)

    assert len(df) == 30
    assert "rolling_win_rate" in df.columns
    assert "rolling_reward" in df.columns
    # Frequency agent learns rock quickly and counters with paper, leading to high rolling win rate
    assert df["rolling_win_rate"].iloc[-1] >= 0.8


def test_summarize_variability_and_confidence_intervals():
    results = run_experiments(rounds=20, seeds=3, switch_round=10)
    summary = summarize(results)

    required_cols = {
        "opponent", "agent", "mean_win_rate", "std_win_rate", "ci95_win_rate",
        "mean_reward", "std_reward", "ci95_reward",
        "mean_prediction_accuracy", "std_prediction_accuracy", "ci95_prediction_accuracy", "seeds"
    }
    assert required_cols.issubset(set(summary.columns))
    assert (summary["seeds"] == 3).all()
    # Confidence interval and std should be non-negative numbers
    assert (summary["std_win_rate"] >= 0).all()
    assert (summary["ci95_win_rate"] >= 0).all()


def test_validate_experiment_fairness():
    results = run_experiments(rounds=15, seeds=2, switch_round=8)
    validation = validate_experiment_fairness(results)

    assert validation["valid"] is True
    assert validation["rounds_per_match"] == 15
    assert validation["seeds_count"] == 2
    assert len(validation["errors"]) == 0

    # Test failure detection if data is unbalanced
    unbalanced = results.iloc[:-5]  # drop some rounds
    bad_val = validate_experiment_fairness(unbalanced)
    assert bad_val["valid"] is False
    assert len(bad_val["errors"]) > 0


def test_evaluate_adaptation_and_summary():
    results = run_experiments(rounds=40, seeds=2, switch_round=20)
    adapt_df = evaluate_adaptation(results, switch_round=20, target_win_rate=0.40)

    assert not adapt_df.empty
    expected_cols = {
        "agent", "seed", "pre_switch_win_rate", "post_switch_win_rate",
        "win_rate_drop", "pre_switch_reward", "post_switch_reward",
        "pre_switch_accuracy", "post_switch_accuracy", "adaptation_latency"
    }
    assert expected_cols.issubset(set(adapt_df.columns))

    summary = summarize_adaptation(adapt_df)
    assert not summary.empty
    assert "mean_pre_win_rate" in summary.columns
    assert "mean_post_win_rate" in summary.columns
    assert "adaptation_success_rate" in summary.columns


def test_export_results(tmp_path: Path):
    results = run_experiments(rounds=10, seeds=1, switch_round=5)
    files = export_results(results, output_dir=tmp_path, prefix="test_run")

    assert files["raw_csv"].exists()
    assert files["summary_csv"].exists()
    assert files["raw_csv"].stat().st_size > 0
    assert files["summary_csv"].stat().st_size > 0


def test_edge_cases():
    agent = RandomAgent(seed=1)
    opponent = RandomOpponent(seed=1)

    with pytest.raises(ValueError, match="rounds must be at least 1"):
        run_match(agent, opponent, rounds=0, seed=1, opponent_name="R", agent_name="R")

    with pytest.raises(ValueError, match="rolling_window must be at least 1"):
        run_match(agent, opponent, rounds=10, seed=1, opponent_name="R", agent_name="R", rolling_window=0)

    # 1 round match should work gracefully
    single_round = run_match(agent, opponent, rounds=1, seed=1, opponent_name="R", agent_name="R")
    assert len(single_round) == 1
    assert single_round["round"].iloc[0] == 1
