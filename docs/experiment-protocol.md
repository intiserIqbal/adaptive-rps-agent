# Experiment Protocol: Adaptive Rock-Paper-Scissors Agent

## 1. Overview and Research Questions

This experiment investigates online opponent modeling in Rock-Paper-Scissors (RPS), evaluating how well different learning agents predict and exploit stationary and non-stationary patterns.

### Core Research Questions
1. **Stationary Exploitation:** Can frequency-based or first-order Markov opponent models reliably outperform a uniform random baseline when faced with stationary non-uniform opponent policies (e.g., rock-biased or cyclic sequences)?
2. **Prediction Accuracy vs. Outcome:** Does a higher prediction accuracy consistently translate into a higher win rate and reward across distinct opponent dynamics?
3. **Adaptation Under Strategy Shifts:** When an opponent suddenly alters its strategy midway through an encounter, how rapidly and effectively do the respective models detect the shift and recover exploitative performance?

---

## 2. Formal Hypotheses

* **Hypothesis 1 (Stationary Bias):** Against a biased opponent with a stationary move distribution (e.g., 70% Rock), the **Frequency Model** will converge to counter the dominant move, yielding an expected win rate significantly exceeding 33.3% and approaching the theoretical optimum (70%).
* **Hypothesis 2 (Sequential/Cyclic Patterns):** Against a deterministic cyclic opponent ($R \to P \to S \to R$), the **First-Order Markov Model** will achieve near 100% prediction accuracy and win rate once the transition matrix is populated, whereas the Frequency Model will fail to outperform chance (~33.3%) because marginal move frequencies remain uniform.
* **Hypothesis 3 (Adaptation Latency):** In a non-stationary strategy-switch scenario (shifting from Rock-heavy to Scissors-heavy at round $K$), both learning models will experience an immediate drop in win rate and reward (switch cost), followed by an adaptation phase whose latency depends on historical inertia.
* **Hypothesis 4 (Random Opponent Baseline):** Against an independent, identically distributed (i.i.d.) uniform random opponent, no model can achieve a long-term expected win rate or prediction accuracy exceeding $1/3 \approx 33.33\%$.

---

## 3. Evaluation Setup and Opponent Scenarios

All agents are evaluated across four standardized opponent policies:

| Opponent Scenario | Description / Policy Distribution | Theoretical Exploitable Feature |
| :--- | :--- | :--- |
| **Random** | Uniform i.i.d. random selection: $P(R)=P(P)=P(S)=\frac{1}{3}$. | None (Nash equilibrium). Baseline sanity check. |
| **Biased (Rock-heavy)** | Fixed stationary distribution: $P(R)=0.7, P(P)=0.2, P(S)=0.1$. | Marginal frequency bias. |
| **Cyclic Pattern** | Strict transition order: $R \to P \to S \to R \dots$ | First-order conditional transition probability. |
| **Strategy Switch** | Round $1 \dots K$: Rock-heavy $(0.7, 0.2, 0.1)$.<br>Round $K+1 \dots N$: Scissors-heavy $(0.1, 0.2, 0.7)$. | Non-stationary distribution switch at round $K$. |

---

## 4. Agents Evaluated

1. **Random Baseline (`RandomAgent`):** Selects moves uniformly at random. Produces no explicit move predictions (`prediction = None`); its prediction accuracy is treated as undefined, not zero.
2. **Frequency Model (`FrequencyAgent`):** Maintains cumulative counts of observed opponent moves and counters the most frequent move $\operatorname{argmax}_{m} C(m)$, with ties broken uniformly at random.
3. **First-Order Markov Model (`MarkovAgent`):** Tracks transition frequencies $C(m_{t-1}, m_t)$. Conditioned on the opponent's previous move $m_{t-1}$, it predicts $\operatorname{argmax}_{m} C(m_{t-1}, m)$ and selects the corresponding counter-move. Falls back to global frequencies when no transition history exists.

---

## 5. Prediction Timing and Fairness (No Data Leakage)

To maintain experimental validity:
* In round $t$, the agent receives `history = [m_1, ..., m_{t-1}]` and must commit to its action $a_t$ and prediction $\hat{m}_t$ **before** the opponent's current move $m_t$ is revealed.
* The opponent generates $m_t$.
* The round outcome is calculated: $\operatorname{outcome}(a_t, m_t) \in \{\text{win}, \text{draw}, \text{loss}\}$.
* Only after the outcome is settled does the agent receive $m_t$ via `agent.observe(m_t)`.
* Every agent-opponent pair is evaluated for identical round counts across deterministic seed schedules.

---

## 6. Metric Definitions

### 6.1 Game Outcome Metrics
* **Win Rate:**
  $$\text{Win Rate}_t = \frac{\sum_{i=1}^t \mathbb{I}(\text{outcome}_i = \text{win})}{t}$$
* **Average Reward:**
  $$\text{Reward}_t = \frac{\sum_{i=1}^t R_i}{t}, \quad \text{where } R_i = \begin{cases} +1 & \text{if win} \\ 0 & \text{if draw} \\ -1 & \text{if loss} \end{cases}$$
* **Rolling Win Rate ($W_k$):**
  $$\text{Rolling Win Rate}_t = \frac{\text{Wins in rounds } [t-k+1, t]}{\min(t, k)}$$

### 6.2 Prediction Metrics
* **Prediction Accuracy:** Calculated **strictly** on rounds where the model produced an explicit prediction:
  $$\text{Accuracy}_t = \frac{\sum_{i \in \mathcal{P}_t} \mathbb{I}(\hat{m}_i = m_i)}{|\mathcal{P}_t|}$$
  *Note:* The random baseline does not predict; its accuracy is reported as `NaN`.

### 6.3 Adaptation Metrics (for Strategy Switching)
* **Pre-Switch Win Rate:** Win rate for rounds $r \le K$.
* **Post-Switch Win Rate:** Win rate for rounds $r > K$.
* **Win Rate Drop:** $\Delta = \text{Win Rate}_{\text{pre}} - \text{Win Rate}_{\text{post}}$.
* **Adaptation Latency:** Number of rounds following switch ($r - K$) until the agent's rolling win rate recovers to $\ge \tau$ (default recovery threshold $\tau = 0.50$). If unreached, latency is undefined (`NaN`).

### 6.4 Aggregation Across Seeds
* **Mean Performance:** $\bar{x} = \frac{1}{S} \sum_{s=1}^S x_s$
* **Sample Standard Deviation:** $s = \sqrt{\frac{1}{S-1} \sum_{s=1}^S (x_s - \bar{x})^2}$
* **95% Confidence Interval (Half-width):**
  $$\text{CI}_{95} = 1.96 \cdot \frac{s}{\sqrt{S}}$$

---

## 7. Reproduction Steps

### Running Tests
```powershell
python -m pytest
```

### Running Experiments via Python Script
```python
from adaptive_rps.experiment import run_experiments, summarize, evaluate_adaptation, export_results

# Execute reproducible multi-seed evaluation
results = run_experiments(rounds=300, seeds=5, switch_round=150)

# Generate aggregated summary with confidence intervals
summary = summarize(results)
print(summary)

# Evaluate non-stationary adaptation metrics
adaptation = evaluate_adaptation(results, switch_round=150, target_win_rate=0.50)
print(adaptation)

# Export machine-readable CSVs
export_results(results, output_dir="results", prefix="run_300r_5s")
```

### Launching the Interactive Dashboard
```powershell
streamlit run app.py
```
