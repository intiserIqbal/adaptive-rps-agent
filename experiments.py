"""
experiments.py  --  runs every predictor against every bot and saves
    results_summary.csv   (win/draw/loss % for each pair)
    results_curves.csv    (win rate over time, raw numbers for your own plots)
    win_rate_over_time.png  (the main graph for your report)

Run it with:   python experiments.py
"""

import csv
import random

from game import result
from predictors import RandomPredictor, FrequencyPredictor, MarkovPredictor
from bots import ALL_BOTS

# ---- settings you can change -------------------------------------------
ROUNDS = 500      # rounds per match
RUNS = 30         # repeat each match this many times and average (reduces luck)
BLOCK = 50        # win rate is measured over blocks of this many rounds
SEED = 42         # fixed seed so your results are repeatable
# ------------------------------------------------------------------------

PREDICTORS = [RandomPredictor, FrequencyPredictor, MarkovPredictor]


def run_match(predictor_cls, bot_cls, rounds):
    """Play one match. Returns a list of results: +1 win, 0 draw, -1 loss."""
    ai = predictor_cls()
    bot = bot_cls()
    ai_last = None
    outcomes = []
    for _ in range(rounds):
        ai_move = ai.choose_move()           # AI decides first...
        opp_move = bot.next_move(ai_last)    # ...the bot decides independently
        outcomes.append(result(ai_move, opp_move))
        ai.update(opp_move)                  # AI learns from what the bot played
        ai_last = ai_move
    return outcomes


def main():
    random.seed(SEED)
    n_blocks = ROUNDS // BLOCK
    summary_rows = []
    curve_rows = []
    curves = {}  # (bot name, predictor name) -> list of win rates per block

    for bot_cls in ALL_BOTS:
        for pred_cls in PREDICTORS:
            wins = draws = losses = 0
            block_wins = [0] * n_blocks
            for _ in range(RUNS):
                outcomes = run_match(pred_cls, bot_cls, ROUNDS)
                for i, r in enumerate(outcomes):
                    if r == 1:
                        wins += 1
                        block_wins[i // BLOCK] += 1
                    elif r == 0:
                        draws += 1
                    else:
                        losses += 1
            total = RUNS * ROUNDS
            bot_name, pred_name = bot_cls.name, pred_cls.name
            summary_rows.append([bot_name, pred_name,
                                 round(100 * wins / total, 1),
                                 round(100 * draws / total, 1),
                                 round(100 * losses / total, 1)])
            rates = [100 * bw / (RUNS * BLOCK) for bw in block_wins]
            curves[(bot_name, pred_name)] = rates
            for b, rate in enumerate(rates):
                curve_rows.append([bot_name, pred_name, (b + 1) * BLOCK, round(rate, 1)])

    # ---- save CSV files ----
    with open("results_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opponent", "AI", "win_%", "draw_%", "loss_%"])
        w.writerows(summary_rows)
    with open("results_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opponent", "AI", "round", "win_rate_%_in_last_block"])
        w.writerows(curve_rows)

    # ---- print a table ----
    print(f"\nResults: {RUNS} runs x {ROUNDS} rounds per pair\n")
    print(f"{'Opponent':<28}{'AI':<20}{'Win%':>6}{'Draw%':>7}{'Loss%':>7}")
    print("-" * 68)
    for row in summary_rows:
        print(f"{row[0]:<28}{row[1]:<20}{row[2]:>6}{row[3]:>7}{row[4]:>7}")

    # ---- draw the graph ----
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed - skipping graph. Run: pip install matplotlib")
        return

    bots = [b.name for b in ALL_BOTS]
    preds = [p.name for p in PREDICTORS]
    x = [(i + 1) * BLOCK for i in range(n_blocks)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()
    for ax, bot_name in zip(axes, bots):
        for pred_name in preds:
            ax.plot(x, curves[(bot_name, pred_name)], marker="o", label=pred_name)
        ax.axhline(100 / 3, color="gray", linestyle="--", linewidth=1)  # chance level
        ax.set_title(f"vs {bot_name}")
        ax.set_xlabel("Round")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("AI win rate (%)")
    axes[3].set_ylabel("AI win rate (%)")
    axes[0].legend()
    for ax in axes[len(bots):]:
        ax.axis("off")  # hide the unused 6th panel
    fig.suptitle("AI win rate over time (dashed line = 33% chance level)")
    fig.tight_layout()
    fig.savefig("win_rate_over_time.png", dpi=150)
    print("\nSaved: results_summary.csv, results_curves.csv, win_rate_over_time.png")


if __name__ == "__main__":
    main()
