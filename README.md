# Adaptive RPS Agent

A small, reproducible research demo comparing online opponent models in Rock-Paper-Scissors.

## Research question

How do frequency-based and first-order Markov opponent models compare in predicting and exploiting patterns under stationary and changing opponent strategies?

## Included

- Random baseline, frequency model, and first-order Markov model
- Controlled random, biased, cyclic, and strategy-switching opponents
- Repeated experiments with deterministic seeds
- Round-level CSV export, summary table, and win-rate/prediction-accuracy charts
- Unit tests and GitHub Actions CI
- Basic Streamlit dashboard

## Quick start (Python 3.11+)

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
streamlit run app.py
```

If PowerShell blocks activation, run the environment's Python directly instead or use Command Prompt activation (`.venv\Scripts\activate.bat`). **Do not weaken system-wide execution policy just for this project.**

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
streamlit run app.py
```

## Experimental design

Each agent chooses before the current opponent move is revealed. It receives the observed opponent move only after the round, preventing direct access to the answer. Every combination of agent, opponent type, and seed is evaluated for the same number of rounds.

- **Random baseline:** selects uniformly at random; it does not make explicit predictions.
- **Frequency model:** counters the most frequently observed opponent move.
- **First-order Markov model:** estimates the next move from the previous opponent move, falling back to global frequencies when there is no transition history.
- **Opponents:** uniform random, fixed rock-heavy distribution, repeating rock-paper-scissors cycle, and a distribution that changes halfway through a match.

Metrics include cumulative win rate, average reward (`win=+1`, `draw=0`, `loss=-1`), and prediction accuracy on rounds where a prediction is available. Prediction accuracy and win rate measure different things.

## Important limitations

These are baseline models, not state-of-the-art AI. Results depend on opponent behavior and random variation. A genuinely random opponent has no exploitable historical pattern in expectation. Do not claim a universal win rate or claim that the models generalize to all human players. Run more seeds and report variability before making research conclusions.

## Team workflow

1. Create feature branches for individual changes.
2. Open pull requests into `main`.
3. Have at least one teammate review each pull request.
4. Let CI run `pytest` and `ruff check .` before merging.
5. Record experiment settings and results; do not commit generated result files unless needed for the report.

## Next improvements

- Add confidence intervals across independent seeds.
- Add a human-play mode and tests for the round flow.
- Add a formal adaptation-time metric after strategy switches.
- Consider Bayesian smoothing and probabilistic action selection.
- Add webcam recognition only after the research demo is stable.
