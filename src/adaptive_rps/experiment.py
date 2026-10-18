"""Simulation and metrics for comparing agents against controlled opponents."""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from adaptive_rps.game import Move, outcome
from adaptive_rps.models import FrequencyAgent, MarkovAgent, RandomAgent
from adaptive_rps.opponents import BiasedOpponent, CycleOpponent, RandomOpponent, SwitchingOpponent

AgentFactory = Callable[[int], object]
OpponentFactory = Callable[[int, int], object]


def run_match(agent, opponent, rounds: int, seed: int, opponent_name: str, agent_name: str) -> pd.DataFrame:
    """Run a match; the agent must choose before the opponent's current move is revealed."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    history: list[Move] = []
    rows: list[dict[str, object]] = []
    wins = losses = draws = correct = predictions = 0
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
        history.append(opponent_move)
        agent.observe(opponent_move)
        rows.append({
            "seed": seed, "opponent": opponent_name, "agent": agent_name,
            "round": round_number, "agent_move": agent_move,
            "opponent_move": opponent_move, "outcome": result,
            "win_rate": wins / round_number,
            "prediction_accuracy": correct / predictions if predictions else float("nan"),
            "wins": wins, "losses": losses, "draws": draws,
            "average_reward": (wins - losses) / round_number,
        })
    return pd.DataFrame(rows)


def run_experiments(rounds: int = 300, seeds: int = 5, switch_round: int = 150) -> pd.DataFrame:
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
                frames.append(run_match(agent, opponent, rounds, seed, opponent_name, agent_name))
    return pd.concat(frames, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize final per-match performance across seeds."""
    final_rows = results.sort_values("round").groupby(["seed", "opponent", "agent"], as_index=False).tail(1)
    return (final_rows.groupby(["opponent", "agent"], as_index=False)
            .agg(mean_win_rate=("win_rate", "mean"), mean_reward=("average_reward", "mean"),
                 mean_prediction_accuracy=("prediction_accuracy", "mean"),
                 seeds=("seed", "nunique")))
