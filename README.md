# Adaptive Rock-Paper-Scissors AI: Complete Guide (CSE440)

An AI that learns how you play Rock-Paper-Scissors, predicts your next move, and plays the move that beats it. This guide covers setup from scratch, what every file does, how to run it, and how to use the results in your report and presentation.

---

## 1. Setup from scratch

1. **Install Python 3.9 or newer** from python.org. On Windows, tick **"Add Python to PATH"** during installation.
2. **Install matplotlib** (only needed for the graph). Open a terminal (Command Prompt on Windows, Terminal on Mac) and run:
   ```
   pip install matplotlib
   ```
3. **Put all the files in one folder**, for example `rps_ai/`:
   ```
   rps_ai/
     game.py
     predictors.py
     bots.py
     experiments.py
     play.py
   ```
   They must all be in the *same* folder, because they import each other.
4. **Open a terminal inside that folder.** On Windows, open the folder in File Explorer, click the address bar, type `cmd` and press Enter. Or open the folder in VS Code and use its built-in terminal.
5. If `python` doesn't work, try `python3` (Mac/Linux) or `py` (Windows).

## 2. How to run it

| Command | What it does | When you use it |
|---|---|---|
| `python play.py` | You play against the AI in the terminal | Live demo, and to check the AI learns you |
| `python experiments.py` | Runs every AI against every bot, prints a results table, saves 2 CSV files and 1 graph | Producing results for the report |

`experiments.py` takes only a few seconds. It creates `results_summary.csv`, `results_curves.csv` and `win_rate_over_time.png` in the same folder.

## 3. What each file does

**`game.py` (the environment).** Defines the moves `R`, `P`, `S`, the `BEATS` dictionary (`BEATS["R"] = "P"` means Paper beats Rock), and `result(ai_move, opp_move)`, which returns +1 (AI wins), 0 (draw) or -1 (AI loses). This is the "environment design" part of your report.

**`predictors.py` (the AI).** Three classes, all with the same interface:
- `predict()` returns the opponent's most likely next move, or `None` if there is no data yet.
- `update(opp_move)` learns from the move the opponent just played.
- `choose_move()` returns the AI's move: the counter to the prediction, or a random move if there is no prediction yet.

The three predictors:
- **RandomPredictor**: never learns. It is a control group: if your real AIs don't beat this, they aren't learning.
- **FrequencyPredictor (Algorithm A)**: keeps a count of each move the opponent has played, predicts the most common one, and plays its counter.
- **MarkovPredictor (Algorithm B)**: keeps a 3x3 table of "what did the opponent play *after* each previous move?"

**`bots.py` (simulated opponents).** Five bots with known patterns: Random, Biased (50% Rock), Cyclic (R,P,S,R,P,S...), Switcher (never repeats its last move) and Reactive (plays what beats the AI's last move). Using bots makes your experiments repeatable, and each bot tests a different kind of pattern.

**`experiments.py` (the evaluation).** Plays each AI against each bot for 500 rounds, repeats that 30 times, and averages, so one lucky streak doesn't distort the results. It measures win, draw and loss percentages and the win rate over time in blocks of 50 rounds.

**`play.py` (the demo).** Lets a human play. After every round it shows what the AI guessed, what it played, and the running score. At the end it shows the win rate for the first half versus the second half of the game, which is evidence that the AI learned.

## 4. How the two algorithms work (with an example)

Suppose the opponent's moves so far are: `R, P, R, P, R, P, R`

**Frequency:** counts are R=4, P=3, S=0. The most common move is R, so it predicts R and plays **Paper**.

**Markov:** the opponent's last move was R. The table row for "after R" says the opponent played P every time (3 times). So it predicts **P** and plays **Scissors**. Markov noticed the alternating pattern, and Frequency couldn't.

In the first round, and after any move that has never been seen before, there is no data, so the AI plays randomly.

## 5. Results from a test run

Win % of the AI, averaged over 30 runs of 500 rounds each (yours will be very close; small differences are normal):

| Opponent | Random AI | Frequency | Markov |
|---|---|---|---|
| Random | 32.9 | 33.5 | 33.3 |
| Biased (50% Rock) | 33.1 | 50.1 | 48.2 |
| Cyclic (R,P,S) | 33.2 | 10.9 | 99.5 |
| Switcher (never repeats) | 32.9 | 31.2 | 50.0 |
| Reactive (beats AI's last) | 32.3 | 22.3 | 47.7 |

What this shows, in plain words:
- **Against a random player, nobody can win.** All three stay at about 33%. This is the expected and honest result. Mention it in your report.
- **Biased player:** both learning AIs reach about 50%, well above chance. Frequency is at least as good as Markov here, because a bias is exactly what counting detects.
- **Cyclic player:** Markov wins about 99.5% of rounds because after each move the next one is completely predictable. Frequency actually does *worse* than chance (10.9%), because all three moves look equally common, so its "prediction" is always a guess and its counters lose to the cycle.
- **Switcher:** only Markov (50%) can exploit "never repeats", while Frequency stays at chance.
- **Reactive player:** this is the limitation case. Its move depends on *the AI's* last move, not on its own history, so neither predictor is built to see the pattern. Frequency gets beaten badly (22.3% wins vs 77.4% losses) and Markov only breaks about even (47.7% wins vs 49.1% losses). Use this in your limitations section: both predictors only look at the opponent's history, and a smarter version could also use the AI's own previous move.

The learning is fast: against most bots the AI reaches its final win rate within the first 50 rounds, so the graph curves look flat after the first point. If you want to show a more visible "learning curve", set `BLOCK = 10` in `experiments.py`.

## 6. How to use this in your report and presentation

**Report outline**
1. **Introduction**: the problem, and why Rock-Paper-Scissors is a good test of learning from experience.
2. **Environment design**: moves, scoring, game loop (from `game.py`).
3. **Algorithms**: Frequency and Markov, using the example in section 4.
4. **Experiment setup**: the 5 bots, 500 rounds, 30 runs, metrics (win/draw/loss %, win rate over time).
5. **Results**: paste the table from `results_summary.csv` and the graph `win_rate_over_time.png`.
6. **Discussion**: which algorithm wins against which pattern and why (section 5).
7. **Limitations and future work**: can't beat a random player; the Reactive bot; possible extensions (section 8).

**Presentation and demo**
- Show the graph first, then run `python play.py` live and let a classmate play 20 to 30 rounds with the Markov AI. Deliberately use a pattern (like R, P, S repeated) and let the class watch the AI's win rate rise. Then ask them to try to be random.

## 7. Suggested split for teammates

| Member | Owns | Files |
|---|---|---|
| 1 | Environment and the human demo | `game.py`, `play.py` |
| 2 | The AI algorithms | `predictors.py` |
| 3 | Bots and experiments | `bots.py`, `experiments.py` |
| 4 | Graphs, report and slides | results files and the report |

Everyone should still be able to explain the whole project, because instructors often ask any member about any part.

## 8. Settings you can change and easy extensions

**In `experiments.py`:** `ROUNDS`, `RUNS`, `BLOCK` and `SEED` (top of the file).

**In `bots.py`:** change the Biased bot's `weights=[0.5, 0.25, 0.25]` to make the bias stronger or weaker.

**Extensions if you have spare time (all optional):**
- **Second-order Markov**: predict from the opponent's last *two* moves (table keyed by pairs).
- **Recency weighting**: multiply old counts by 0.9 each round so the AI adapts faster when the opponent changes strategy.
- **AI-aware Markov**: also condition on the AI's own last move, which is what would beat the Reactive bot.
- **A simple GUI** with three buttons instead of typing.

## 9. Troubleshooting

- **`ModuleNotFoundError: No module named 'game'`**: you are not running from the project folder. Open the terminal inside that folder.
- **`ModuleNotFoundError: No module named 'matplotlib'`**: run `pip install matplotlib`. The experiments still print the table and save the CSVs without it.
- **`python` not recognized**: try `python3` or `py`, or reinstall Python with "Add to PATH" ticked.
- **The play script ends immediately**: run it in a real terminal, not by double-clicking the file.
