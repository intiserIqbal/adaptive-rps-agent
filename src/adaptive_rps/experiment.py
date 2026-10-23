"""Simulation and metrics for comparing agents against controlled opponents."""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_rps.game import Move, outcome
from adaptive_rps.models import FrequencyAgent, MarkovAgent, RandomAgent
from adaptive_rps.opponents import BiasedOpponent, CycleOpponent, RandomOpponent, SwitchingOpponent

AgentFactory = Callable[[int], object]
OpponentFactory = Callable[[int, int], object]


def run_match(
    agent,
    opponent,
    rounds: int,
    seed: int,
    opponent_name: str,
    agent_name: str,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """Run a match; the agent must choose before the opponent's current move is revealed.
    
    Includes round-by-round cumulative outcomes as well as rolling window performance
    to monitor transient dynamics and adaptation behavior.
    """
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1")

    history: list[Move] = []
    rows: list[dict[str, object]] = []
    wins = losses = draws = correct = predictions = 0
    recent_outcomes: list[str] = []

    for round_number in range(1, rounds + 1):
        agent_move, predicted = agent.choose(history)
        opponent_move = opponent.play(history)
        result = outcome(agent_move, opponent_move)

        if predicted is not None:
            predictions += 1
            correct += int(predicted == opponent_move)

        wins += int(result == "win")
        losses += int(result == "loss")
        draws += int(result == "draw")

        recent_outcomes.append(result)
        if len(recent_outcomes) > rolling_window:
            recent_outcomes.pop(0)

        rolling_len = len(recent_outcomes)
        rolling_wins = recent_outcomes.count("win")
        rolling_losses = recent_outcomes.count("loss")
        rolling_win_rate = rolling_wins / rolling_len
        rolling_reward = (rolling_wins - rolling_losses) / rolling_len

        history.append(opponent_move)
        agent.observe(opponent_move)

        rows.append({
            "seed": seed,
            "opponent": opponent_name,
            "agent": agent_name,
            "round": round_number,
            "agent_move": agent_move,
            "opponent_move": opponent_move,
            "outcome": result,
            "win_rate": wins / round_number,
            "prediction_accuracy": correct / predictions if predictions else float("nan"),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "average_reward": (wins - losses) / round_number,
            "rolling_win_rate": rolling_win_rate,
            "rolling_reward": rolling_reward,
        })
    return pd.DataFrame(rows)


def run_experiments(
    rounds: int = 300,
    seeds: int = 5,
    switch_round: int = 150,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """Compare agents across opponent types and independent seeds."""
    agent_factories = [
        ("Random baseline", lambda seed: RandomAgent(seed)),
        ("Frequency model", lambda seed: FrequencyAgent(seed)),
        ("First-order Markov model", lambda seed: MarkovAgent(seed)),
    ]
    opponent_factories = [
        ("Random", lambda seed: RandomOpponent(seed)),
        ("Biased (rock-heavy)", lambda seed: BiasedOpponent(seed)),
        ("Cyclic pattern", lambda seed: CycleOpponent()),
        ("Strategy switch", lambda seed: SwitchingOpponent(seed, switch_round)),
    ]
    frames = []
    for seed in range(seeds):
        for opponent_name, opponent_factory in opponent_factories:
            for agent_name, agent_factory in agent_factories:
                agent = agent_factory(seed * 100 + 11)
                opponent = opponent_factory(seed * 100 + 29)
                frames.append(
                    run_match(
                        agent,
                        opponent,
                        rounds,
                        seed,
                        opponent_name,
                        agent_name,
                        rolling_window=rolling_window,
                    )
                )
    return pd.concat(frames, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize final per-match performance across seeds.
    
    Computes mean performance along with standard deviations and 95% confidence
    intervals across independent seeds to capture variability.
    """
    final_rows = results.sort_values("round").groupby(["seed", "opponent", "agent"], as_index=False).tail(1)

    grouped = final_rows.groupby(["opponent", "agent"], as_index=False)
    summary = grouped.agg(
        mean_win_rate=("win_rate", "mean"),
        std_win_rate=("win_rate", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
        mean_reward=("average_reward", "mean"),
        std_reward=("average_reward", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
        mean_prediction_accuracy=("prediction_accuracy", "mean"),
        std_prediction_accuracy=("prediction_accuracy", lambda s: float(s.std(ddof=1)) if len(s.dropna()) > 1 else 0.0),
        seeds=("seed", "nunique"),
    )

    # 95% Confidence Interval half-width: 1.96 * (std / sqrt(N))
    n = summary["seeds"].astype(float)
    summary["ci95_win_rate"] = np.where(
        n > 1,
        1.96 * (summary["std_win_rate"] / np.sqrt(n)),
        0.0,
    )
    summary["ci95_reward"] = np.where(
        n > 1,
        1.96 * (summary["std_reward"] / np.sqrt(n)),
        0.0,
    )
    summary["ci95_prediction_accuracy"] = np.where(
        (n > 1) & summary["mean_prediction_accuracy"].notna(),
        1.96 * (summary["std_prediction_accuracy"] / np.sqrt(n)),
        float("nan"),
    )
    return summary


def evaluate_adaptation(
    results: pd.DataFrame,
    switch_round: int = 150,
    target_win_rate: float = 0.50,
) -> pd.DataFrame:
    """Evaluate agent adaptation behavior before and after strategy switch.
    
    Calculates:
      - pre_switch_win_rate: win rate up to switch_round
      - post_switch_win_rate: win rate strictly after switch_round
      - pre_switch_accuracy / post_switch_accuracy: prediction accuracy in each regime
      - pre_switch_reward / post_switch_reward: average reward in each regime
      - adaptation_latency: number of rounds after switch until rolling win rate
        first reaches or exceeds target_win_rate. If target is never reached,
        adaptation_latency is float('nan').
    """
    if "opponent" in results.columns and "Strategy switch" in results["opponent"].values:
        eval_data = results[results["opponent"] == "Strategy switch"]
    else:
        eval_data = results

    records: list[dict[str, object]] = []
    groups = eval_data.groupby(["agent", "seed"])

    for (agent_name, seed), group in groups:
        sorted_group = group.sort_values("round")
        pre_group = sorted_group[sorted_group["round"] <= switch_round]
        post_group = sorted_group[sorted_group["round"] > switch_round]

        pre_rounds = len(pre_group)
        post_rounds = len(post_group)

        pre_win_rate = pre_group["outcome"].eq("win").sum() / pre_rounds if pre_rounds > 0 else float("nan")
        post_win_rate = post_group["outcome"].eq("win").sum() / post_rounds if post_rounds > 0 else float("nan")

        pre_reward = (
            (pre_group["outcome"].eq("win").sum() - pre_group["outcome"].eq("loss").sum()) / pre_rounds
            if pre_rounds > 0 else float("nan")
        )
        post_reward = (
            (post_group["outcome"].eq("win").sum() - post_group["outcome"].eq("loss").sum()) / post_rounds
            if post_rounds > 0 else float("nan")
        )

        # Prediction accuracy (only defined when predicted moves are made)
        pre_pred_rows = pre_group[pre_group["prediction_accuracy"].notna()]
        pre_accuracy = pre_pred_rows["prediction_accuracy"].iloc[-1] if not pre_pred_rows.empty else float("nan")

        if "prediction_accuracy" in post_group.columns and post_group["prediction_accuracy"].notna().any():
            post_accuracy = post_group["prediction_accuracy"].iloc[-1]
        else:
            post_accuracy = float("nan")

        # Adaptation latency: rounds after switch_round until rolling_win_rate >= target_win_rate
        latency = float("nan")
        for _, row in post_group.iterrows():
            if "rolling_win_rate" in row and row["rolling_win_rate"] >= target_win_rate:
                latency = float(row["round"] - switch_round)
                break

        records.append({
            "agent": agent_name,
            "seed": seed,
            "pre_switch_win_rate": pre_win_rate,
            "post_switch_win_rate": post_win_rate,
            "win_rate_drop": pre_win_rate - post_win_rate if not math.isnan(pre_win_rate) and not math.isnan(post_win_rate) else float("nan"),
            "pre_switch_reward": pre_reward,
            "post_switch_reward": post_reward,
            "pre_switch_accuracy": pre_accuracy,
            "post_switch_accuracy": post_accuracy,
            "adaptation_latency": latency,
        })

    return pd.DataFrame(records)


def summarize_adaptation(adaptation_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize adaptation metrics by agent across seeds."""
    if adaptation_df.empty:
        return pd.DataFrame()

    def calc_mean_latency(s: pd.Series) -> float:
        valid = s.dropna()
        return float(valid.mean()) if not valid.empty else float("nan")

    def calc_success_rate(s: pd.Series) -> float:
        total = len(s)
        return float(s.notna().sum() / total) if total > 0 else 0.0

    summary = adaptation_df.groupby("agent", as_index=False).agg(
        mean_pre_win_rate=("pre_switch_win_rate", "mean"),
        mean_post_win_rate=("post_switch_win_rate", "mean"),
        mean_win_rate_drop=("win_rate_drop", "mean"),
        mean_pre_reward=("pre_switch_reward", "mean"),
        mean_post_reward=("post_switch_reward", "mean"),
        mean_latency=("adaptation_latency", calc_mean_latency),
        adaptation_success_rate=("adaptation_latency", calc_success_rate),
        seeds=("seed", "nunique"),
    )
    return summary


def validate_experiment_fairness(results: pd.DataFrame) -> dict[str, object]:
    """Validate that experimental runs adhere to fairness standards:
    equal rounds, identical seed schedules across all pairings, and schema consistency.
    """
    required_cols = {"seed", "opponent", "agent", "round", "agent_move", "opponent_move", "outcome", "win_rate"}
    missing = required_cols - set(results.columns)
    if missing:
        return {"valid": False, "errors": [f"Missing required columns: {missing}"]}

    errors: list[str] = []
    # Check rounds per match
    rounds_per_match = results.groupby(["seed", "opponent", "agent"])["round"].count()
    unique_round_counts = rounds_per_match.unique()
    if len(unique_round_counts) > 1:
        errors.append(f"Unequal round counts detected across matches: {unique_round_counts.tolist()}")

    # Check seed schedules
    seeds_per_pairing = results.groupby(["opponent", "agent"])["seed"].nunique()
    if len(seeds_per_pairing.unique()) > 1:
        errors.append(f"Unequal seed counts detected across opponent-agent pairings: {seeds_per_pairing.to_dict()}")

    return {
        "valid": len(errors) == 0,
        "total_matches": len(rounds_per_match),
        "rounds_per_match": int(unique_round_counts[0]) if len(unique_round_counts) == 1 else None,
        "seeds_count": int(seeds_per_pairing.iloc[0]) if not seeds_per_pairing.empty else 0,
        "errors": errors,
    }


def export_results(
    results: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "rps_experiment",
) -> dict[str, Path]:
    """Export raw round results and summary tables to CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results_file = out / f"{prefix}_raw.csv"
    summary_file = out / f"{prefix}_summary.csv"

    results.to_csv(results_file, index=False)
    summary = summarize(results)
    summary.to_csv(summary_file, index=False)

    return {"raw_csv": results_file, "summary_csv": summary_file}
