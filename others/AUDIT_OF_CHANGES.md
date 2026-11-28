# CSE440 Project Audit: Adaptive RPS Agent Refactor & Enhancement

**Project Name:** Adaptive Rock-Paper-Scissors Agent  
**Course:** CSE440: Artificial Intelligence  
**Audit Date:** November 27, 2026  
**Repository Working Directory:** `g:\NSU\CSE440\proj\adaptive-rps-agent`  
**Audit File Location:** `others/AUDIT_OF_CHANGES.md`

---

## 1. Executive Summary

This audit records the repository's development from its initial experiment framework through the current interactive research demo. The chronological summary below is based on the available Git history and the changes in the current working tree.

### Summary of Major Deliverables:
1. **Exponentially Decayed Markov Model (`DecayMarkovAgent`)**: Solved the concept drift problem in non-stationary environments using an exponential discount parameter ($\gamma = 0.90$).
2. **Second-Order Markov Model (`Markov2Agent`)**: Extended sequence learning to conditioning on joint 2-step history $(s_{t-2}, s_{t-1})$ with hierarchical backoff.
3. **Full Streamlit Front-End Redesign (`app.py`)**:
   - **Tab 1: Live Arena (Human vs. AI)** with real-time interactive game loop, explainable AI (XAI) belief distributions, and scoreboards.
   - **Tab 2: Benchmark Experiments** with interactive Altair charts, shaded confidence bands ($\pm 1\sigma$), and strategy switch reference lines.
   - **Tab 3: Model Brain Visualizer** with interactive $3 \times 3$ transition probability heatmaps ($P(M_t \mid M_{t-1})$).
   - **Tab 4: Methodology & Theory** detailing game-theoretic Nash equilibrium and mathematical formulas.
4. **Comprehensive Test Suite & Quality Assurance**: All unit tests expanded and passing (9/9), linter verified clean with `ruff`.
5. **Win-Stay, Lose-Shift opponent**: Added outcome-dependent behavior (repeat after a win/draw, shift after a loss) to the reproducible benchmark.
6. **Dashboard polish**: Refined the visual theme and hero area, made the benchmark default to WSLS, and refreshes benchmark data when simulation settings change.

---

## 2. Development History From the Beginning

The repository history contains the initial setup and feature commits, plus two experiment-protocol pull requests. The WSLS and UI updates are integrated with those remote experiment changes in the current local branch.

| Stage | Change | Result |
| :--- | :--- | :--- |
| Initial commit (`b049bda`) | Established the Adaptive RPS project. | Repository baseline. |
| Initial experiment framework (`5028d7e`) | Added the first game and experiment framework, opponent/model code, and project setup. | Runnable experiment foundation. |
| CI/signing follow-up (`382882b`) | Added a repository workflow change to verify automatic GPG signing. | GitHub/repository workflow support. |
| AI and dashboard expansion (`c907002`) | Added Decayed Markov and Second-order Markov agents, the interactive Streamlit arena and visualizers, plus this audit document. | Expanded from the initial experiment into a multi-model demonstration. |
| Experiment protocol PR (`5483a4f`, merged as PR #2) | Added rolling performance, seed variability, confidence intervals, experiment-fairness validation, adaptation metrics, result export, and the protocol document. | Stronger experiment reporting and reproducibility checks. |
| Adaptation metric fix (`e6dae71`, merged as PR #3) | Measures recovery only on post-switch rounds and requires recovery to persist. | Avoids counting pre-switch performance as immediate adaptation. |
| Current updates (November 27, 2026) | Added WSLS opponent behavior, included it in reproducible benchmarks, improved dashboard styling and benchmark refresh behavior, and restored all five agents alongside the experiment protocol updates. | Outcome-driven opponent and updated experiment metrics are available together. |

### Current change details

- `src/adaptive_rps/opponents.py`: Added `WSLSOpponent`. It starts with a seeded random move, then repeats its previous move after a win or draw and randomly selects one of the other two moves after a loss.
- `src/adaptive_rps/experiment.py`: Preserved the protocol PR's rolling metrics, seed variability, confidence intervals, adaptation-latency tools, fairness validation, and CSV exports. Restored all five agents, enabled the selected-agent filter used by the dashboard, added WSLS to the benchmark set, and call an opponent's `observe(agent_move, opponent_move)` hook after each round when available.
- `docs/experiment-protocol.md`: Retained the experiment design and metric definitions from the protocol PR.
- `app.py`: Added the WSLS description in the benchmark and methodology views, made WSLS the initial opponent in the learning-curve selector, and refreshed the visual theme with gradient panels, metric cards, and more responsive controls. Benchmark results now rerun when rounds, seeds, switch round, or selected agents change.
- `README.md`: Listed WSLS and the UI refresh as completed project features.
- `tests/test_experiment.py`: Kept the expanded protocol regression coverage and updated expected result counts for five agents and five opponents. Tests have not been rerun during this integration.

## 3. Quantitative Git Diff Summary (November 12 audit snapshot)

```text
 README.md                      |  26 +--
 app.py                         | 573 +++++++++++++++++++++++++++++++++++++----
 src/adaptive_rps/experiment.py |  46 ++-
 src/adaptive_rps/models.py     | 165 ++++++++-
 tests/test_experiment.py       |  12 +-
 tests/test_models.py           |  53 ++-
 6 files changed, 797 insertions(+), 78 deletions(-)
```

---

## 4. Detailed File-by-File Audit

### 3.1. `src/adaptive_rps/models.py` (Core Agent Architectures)

#### A. New Class: `DecayMarkovAgent`
- **Purpose:** Addresses the limitation where standard Markov counts accumulate infinitely, rendering the agent sluggish when an opponent dynamically switches strategies (concept drift).
- **Mathematical Update Rule:**
  $$N_t(s, m) = \gamma \cdot N_{t-1}(s, m) + \mathbb{I}(M_{t-1} = s, M_t = m)$$
  $$G_t(m) = \gamma \cdot G_{t-1}(m) + \mathbb{I}(M_t = m)$$
  where $\gamma \in (0, 1)$ is the memory discount factor (default $\gamma = 0.90$).
- **Key Methods Added:**
  - `__init__(seed=0, gamma=0.90)`: Validates $0 < \gamma < 1$, initializes decayed transition tables and global fallback counts.
  - `choose(history)`: Evaluates decayed transition weights from the previous state; breaks ties deterministically via seeded PRNG; falls back to global decayed frequencies or uniform exploration if untracked.
  - `observe(move)`: Decays all historical transition weights by multiplying by $\gamma$, then increments the specific transition observed.
  - `get_transition_probabilities()`: Computes row-normalized conditional probabilities $P(\text{next} \mid \text{current})$ for visual inspection.
  - `get_next_move_probabilities()`: Exposes the agent's real-time belief distribution over the opponent's next move.

#### B. New Class: `Markov2Agent`
- **Purpose:** Extends sequential pattern recognition to second-order temporal dependencies:
  $$P(M_t = m \mid M_{t-1} = s_1, M_{t-2} = s_2)$$
- **Hierarchical Fallback (Backoff Strategy):**
  1. Checks second-order transition history: $(s_{t-2}, s_{t-1}) \to m$
  2. If unseen, falls back to first-order transition: $s_{t-1} \to m$
  3. If unseen, falls back to global move frequency: $m$
  4. If empty (cold start), falls back to uniform random choice.
- **Key Methods Added:**
  - `choose(history)`: Applies hierarchical backoff to predict and select the optimal counter-move.
  - `observe(move)`: Concurrently updates order-2 transitions, order-1 transitions, and global frequencies.
  - `get_next_move_probabilities()`: Returns normalized probability vector across all three moves.

#### C. Enhancements to Existing Agent Classes
- **`RandomAgent`:** Added `get_next_move_probabilities()` returning uniform $\left[\frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right]$.
- **`FrequencyAgent`:** Added `get_next_move_probabilities()` returning normalized empirical historical frequencies.
- **`MarkovAgent`:** Added `get_transition_probabilities()` and `get_next_move_probabilities()`.

---

### 3.2. `src/adaptive_rps/experiment.py` (Simulation & Pipeline)

- **`DEFAULT_AGENT_FACTORIES`**:
  Expanded from 3 to 5 baseline agents:
  1. Random baseline
  2. Frequency model
  3. First-order Markov model
  4. Decayed Markov model ($\gamma = 0.90$)
  5. Second-order Markov model
- **`run_experiments(...)`**:
  - Added `selected_agents: Sequence[str] | None = None` parameter to allow selective benchmarking in both CLI and Streamlit interfaces.
- **`summarize(...)`**:
  - Added `std_win_rate=("win_rate", "std")` aggregation to compute standard deviation across independent Monte Carlo seeds.

---

### 3.3. `app.py` (Streamlit UI/UX & Interactive Engine)

Transformed from a static, 60-line script with Matplotlib into an interactive, multi-tab demonstration platform:

#### Visual Design & Theming
- Injected custom CSS for dark/glassmorphic metric cards (`st.metric`), hover-responsive action buttons, and color-coded status badges (`badge-win`, `badge-loss`, `badge-draw`).
- Structured using top-level `st.tabs`:

#### Tab 1: 🎮 Live Arena (Human vs. AI)
- **Live Match Engine:** Allows the user to select any AI agent and play round-by-round by clicking **✊ Rock**, **✋ Paper**, or **✌️ Scissors**.
- **Scoreboard:** Real-time metrics tracking Rounds, Human Wins, AI Wins, Draws, and AI Win Rate.
- **Explainable AI (XAI) Belief Bar:** Renders an interactive Altair horizontal bar chart showing the AI's internal probability distribution $P(\text{Rock}), P(\text{Paper}), P(\text{Scissors})$ updated live as the human plays.
- **Thought Bubble:** Displays the AI's prediction and the reasoning behind its counter-move.
- **Match History:** Expandable table showing the last 10 rounds with color indicators.
- **Reset Button:** Allows clearing match history and re-initializing the selected agent.

#### Tab 2: 📊 Benchmark Experiments
- Replaced static Matplotlib with **Altair interactive charts**:
  - **Confidence Bands:** Shaded area plots indicating mean win rate $\pm 1\sigma$ across independent seeds.
  - **Strategy Shift Marker:** Red dashed vertical line at $x = \text{switch\_round}$ with a label highlighting the moment of concept drift.
  - **Prediction Accuracy Curves:** Tracking accuracy over time for predicting agents.
  - **CSV Export:** One-click download of all round-level data.

#### Tab 3: 🧠 Model Brain Visualizer (Explainability)
- **$3 \times 3$ Transition Heatmap:** Color-graded matrix displaying $P(\text{Next Move} \mid \text{Previous Move})$.
- Supports inspecting agents from the **Live Arena match**, or pre-trained on **Cyclic** or **Biased** opponents.
- **Conditional Inspector:** Interactive selector enabling the user to query: *"What does the model think if the previous move was Paper?"*, rendering progress bars for each possible next move.

#### Tab 4: 📖 Methodology & Theory
- Mathematical formulation of Nash Equilibrium ($\pi^* = [1/3, 1/3, 1/3]$) and explanation of why no agent can beat true random in expectation.
- Mathematical formulations for Frequency, Markov-1, Markov-2, and Decayed Markov adaptation.

---

### 3.4. `tests/` (Verification & Regression Testing)

- **`tests/test_models.py`**:
  - `test_frequency_agent_learns_fixed_move`: Added assertion verifying belief probability output.
  - `test_markov_agent_learns_transition`: Added assertion verifying transition probability output.
  - `test_decay_markov_agent_adapts_to_shift`: Verified that after 30 observations of (Rock $\to$ Paper) followed by 15 observations of (Rock $\to$ Scissors), the decayed Markov model ($\gamma=0.85$) correctly prioritizes Scissors due to older memory decay.
  - `test_markov2_agent_learns_sequence`: Verified that the second-order model accurately predicts the 3rd element of a sequence given the previous two.
  - `test_random_agent_probabilities`: Verified uniform distribution.
- **`tests/test_experiment.py`**:
  - Updated grid dimensions from $10 \times 1 \times 4 \times 3$ to $10 \times 1 \times 4 \times 5 = 200$ rows.
  - Added `test_experiment_selected_agents` verifying that passing a custom subset of agents filters the experiment matrix correctly.

---

### 3.5. `README.md` (Documentation)

- Updated **Included** section with all 5 agent architectures and the new multi-tab dashboard features.
- Updated **Next Improvements** checklist to mark off completed milestones (confidence intervals, human play, decay Markov, 2nd-order Markov, transition heatmaps).

---

## 5. Agent Model Comparison Matrix

| Agent | Memory Complexity | State Representation $s_t$ | Decision Rule | Non-Stationary Adaptation |
| :--- | :--- | :--- | :--- | :--- |
| **Random Baseline** | $O(1)$ | None | $a_t \sim \text{Uniform}(\mathcal{M})$ | Static (immune to drift) |
| **Frequency Model** | $O(|\mathcal{M}|)$ | None | $\text{counter}(\arg\max_m N(m))$ | None (infinite memory accumulation) |
| **First-Order Markov** | $O(|\mathcal{M}|^2)$ | $M_{t-1}$ | $\text{counter}(\arg\max_m N(M_{t-1}, m))$ | Slow (accumulated counts cause inertia) |
| **Decayed Markov** | $O(|\mathcal{M}|^2)$ | $M_{t-1}$ | $\text{counter}(\arg\max_m N_t(M_{t-1}, m))$ | **Fast ($\gamma=0.90$ decays past by $>95\%$ in 30 rds)** |
| **Second-Order Markov** | $O(|\mathcal{M}|^3)$ | $(M_{t-2}, M_{t-1})$ | $\text{counter}(\arg\max_m N(M_{t-2}, M_{t-1}, m))$ | Hierarchical backoff to order-1 & global |
| **Win-Stay, Lose-Shift opponent** | $O(1)$ | Previous move and previous outcome | Repeat after win/draw; choose another move after loss | Outcome-dependent |

---

## 6. Verification Commands & Execution Status

The checks below were recorded in the November 12 audit and describe that earlier project state:

```powershell
# 1. Run unit tests
pytest
# Result: 9 passed in 0.90s

# 2. Run Ruff linter
ruff check .
# Result: All checks passed!

# 3. Launch application
streamlit run app.py
```

*Audit document prepared and archived in `others/AUDIT_OF_CHANGES.md`.*

For the November 27 updates, the benchmark test expectations were adjusted, but tests, linting, and a UI launch were not run during that update.
