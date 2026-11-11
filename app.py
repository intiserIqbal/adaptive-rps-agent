"""Streamlit interactive demo dashboard for Adaptive RPS Agent (CSE440)."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from adaptive_rps.experiment import DEFAULT_AGENT_FACTORIES, run_experiments, summarize
from adaptive_rps.game import MOVES, Move, outcome
from adaptive_rps.models import (
    DecayMarkovAgent,
    FrequencyAgent,
    Markov2Agent,
    MarkovAgent,
    RandomAgent,
)

st.set_page_config(
    page_title="Adaptive RPS Agent — CSE440 AI Project",
    page_icon="✊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modern CSS styling
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 34, 42, 0.7) 0%, rgba(20, 24, 30, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(66, 153, 225, 0.4);
    }
    /* Move Buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(0, 114, 255, 0.25);
    }
    /* Badges */
    .arena-card {
        padding: 18px 22px;
        border-radius: 12px;
        margin-bottom: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(25, 30, 38, 0.6);
    }
    .badge-win {
        color: #48bb78;
        font-weight: 700;
    }
    .badge-loss {
        color: #f56565;
        font-weight: 700;
    }
    .badge-draw {
        color: #ecc94b;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🤖 Adaptive RPS Agent")
    st.caption("CSE440: Artificial Intelligence Demonstration")
    st.markdown("---")
    st.header("Simulation Settings")
    rounds = st.slider("Rounds per match", min_value=50, max_value=800, value=250, step=50)
    seeds = st.slider("Independent seeds", min_value=1, max_value=20, value=5)
    switch_round = st.slider("Strategy switch round", min_value=10, max_value=rounds - 1, value=min(125, rounds - 1), step=10)

    all_agent_names = [name for name, _ in DEFAULT_AGENT_FACTORIES]
    selected_agents = st.multiselect(
        "Agent models in simulation",
        options=all_agent_names,
        default=all_agent_names,
    )
    run_sim = st.button("🚀 Run Benchmark Simulation", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown(
        "**Key Concept Drift Model:**\n"
        "- **Decayed Markov:** Applies $\\gamma=0.90$ to forget old states and rapidly adapt to non-stationary opponents."
    )

# Hero Header
st.title("Adaptive Rock-Paper-Scissors: Online Opponent Modeling")
st.markdown(
    "A research and demonstration platform comparing **Frequency**, **First-order Markov**, **Second-order Markov**, "
    "and **Decayed Markov** agents under stationary and changing opponent strategies."
)

tab_arena, tab_benchmark, tab_brain, tab_theory = st.tabs([
    "🎮 Live Arena (Human vs. AI)",
    "📊 Benchmark Experiments",
    "🧠 Model Brain Visualizer",
    "📖 Methodology & Theory",
])

MOVE_ICONS: dict[Move, str] = {"rock": "✊ Rock", "paper": "✋ Paper", "scissors": "✌️ Scissors"}


# ---------------------------------------------------------------------------
# TAB 1: LIVE ARENA (HUMAN VS. AI)
# ---------------------------------------------------------------------------
def init_arena_agent(agent_choice: str, seed: int = 42):
    if agent_choice == "Decayed Markov model (γ=0.85)":
        return DecayMarkovAgent(seed=seed, gamma=0.85)
    if agent_choice == "Second-order Markov model":
        return Markov2Agent(seed=seed)
    if agent_choice == "First-order Markov model":
        return MarkovAgent(seed=seed)
    if agent_choice == "Frequency model":
        return FrequencyAgent(seed=seed)
    return RandomAgent(seed=seed)


with tab_arena:
    st.subheader("Play Against the Adaptive AI")
    st.caption("Make your moves. Notice how the adaptive models detect your subconscious patterns over 10–15 rounds!")

    col_model, col_reset = st.columns([3, 1])
    with col_model:
        arena_agent_name = st.selectbox(
            "Choose AI Opponent Model",
            options=[
                "First-order Markov model",
                "Decayed Markov model (γ=0.85)",
                "Second-order Markov model",
                "Frequency model",
                "Random baseline",
            ],
            index=0,
            key="arena_agent_select",
        )
    with col_reset:
        st.write("")
        st.write("")
        if st.button("🔄 Reset Match", use_container_width=True):
            st.session_state.arena_history = []
            st.session_state.arena_agent = init_arena_agent(arena_agent_name)
            st.session_state.last_round_info = None
            st.rerun()

    # Session State Initialization for Arena
    if "arena_history" not in st.session_state:
        st.session_state.arena_history = []
    if "arena_agent" not in st.session_state or st.session_state.get("current_arena_agent_name") != arena_agent_name:
        st.session_state.arena_agent = init_arena_agent(arena_agent_name)
        st.session_state.current_arena_agent_name = arena_agent_name
        st.session_state.arena_history = []
        st.session_state.last_round_info = None

    history = st.session_state.arena_history
    human_wins = sum(1 for h in history if h["result"] == "Human Win")
    ai_wins = sum(1 for h in history if h["result"] == "AI Win")
    draws = sum(1 for h in history if h["result"] == "Draw")
    total_rounds = len(history)
    ai_win_rate = (ai_wins / total_rounds) if total_rounds > 0 else 0.0

    # Scoreboard Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rounds Played", total_rounds)
    m2.metric("👤 Human Wins", human_wins)
    m3.metric("🤖 AI Wins", ai_wins)
    m4.metric("🤝 Draws", draws)
    m5.metric("AI Win Rate", f"{ai_win_rate:.1%}")

    st.markdown("---")

    # Action Buttons
    st.markdown("### Choose your move:")
    b1, b2, b3 = st.columns(3)
    user_move: Move | None = None

    with b1:
        if st.button("✊ Rock", use_container_width=True, type="secondary"):
            user_move = "rock"
    with b2:
        if st.button("✋ Paper", use_container_width=True, type="secondary"):
            user_move = "paper"
    with b3:
        if st.button("✌️ Scissors", use_container_width=True, type="secondary"):
            user_move = "scissors"

    if user_move is not None:
        agent = st.session_state.arena_agent
        obs_history = [h["human_move"] for h in history]

        # Get AI decision BEFORE revealing human move
        ai_move, predicted = agent.choose(obs_history)
        round_outcome = outcome(ai_move, user_move)  # 'win', 'loss', or 'draw' from AI perspective

        if round_outcome == "win":
            result_str = "AI Win"
        elif round_outcome == "loss":
            result_str = "Human Win"
        else:
            result_str = "Draw"

        # AI updates observation
        agent.observe(user_move)

        round_data = {
            "round": total_rounds + 1,
            "human_move": user_move,
            "ai_move": ai_move,
            "predicted": predicted,
            "result": result_str,
        }
        st.session_state.arena_history.append(round_data)
        st.session_state.last_round_info = round_data
        st.rerun()

    # Display Last Round Outcome & AI Explanation
    last_info = st.session_state.get("last_round_info")
    if last_info:
        r_num = last_info["round"]
        h_m = MOVE_ICONS[last_info["human_move"]]
        a_m = MOVE_ICONS[last_info["ai_move"]]
        res = last_info["result"]
        pred = MOVE_ICONS[last_info["predicted"]] if last_info["predicted"] else "None (exploring)"

        if res == "Human Win":
            st.success(f"**Round {r_num}: You Won!** You played {h_m} and AI played {a_m}.")
        elif res == "AI Win":
            st.error(f"**Round {r_num}: AI Won!** AI played {a_m} which counters your {h_m}.")
        else:
            st.warning(f"**Round {r_num}: Draw!** Both played {h_m}.")

        # Model Thought / Belief Bar
        agent = st.session_state.arena_agent
        if hasattr(agent, "get_next_move_probabilities"):
            probs = agent.get_next_move_probabilities()
            prob_df = pd.DataFrame([
                {"Move": "✊ Rock", "Probability": probs.get("rock", 0.0)},
                {"Move": "✋ Paper", "Probability": probs.get("paper", 0.0)},
                {"Move": "✌️ Scissors", "Probability": probs.get("scissors", 0.0)},
            ])

            col_chart, col_exp = st.columns([3, 2])
            with col_chart:
                st.markdown("#### 🧠 AI Belief: Predicted Probability of Your Next Move")
                prob_chart = (
                    alt.Chart(prob_df)
                    .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, height=22)
                    .encode(
                        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
                        y=alt.Y("Move:N", sort=None, title=""),
                        color=alt.Color("Move:N", legend=None, scale=alt.Scale(
                            domain=["✊ Rock", "✋ Paper", "✌️ Scissors"],
                            range=["#3182ce", "#48bb78", "#ed8936"],
                        )),
                        tooltip=["Move", alt.Tooltip("Probability:Q", format=".1%")],
                    )
                    .properties(height=160)
                )
                st.altair_chart(prob_chart, use_container_width=True)

            with col_exp:
                st.markdown("#### 🔍 AI Thought Process")
                st.info(
                    f"**Predicted Human Move:** {pred}\n\n"
                    f"**AI Strategy:** Countered with {a_m}.\n\n"
                    f"*As rounds increase, the Markov transition table converges toward your habitual transition probabilities.*"
                )

    # History Feed
    if history:
        with st.expander("📜 Round History (Last 10 Rounds)", expanded=False):
            disp_history = pd.DataFrame(history[-10:][::-1])
            disp_history["human_move"] = disp_history["human_move"].map(MOVE_ICONS)
            disp_history["ai_move"] = disp_history["ai_move"].map(MOVE_ICONS)
            disp_history["predicted"] = disp_history["predicted"].map(lambda p: MOVE_ICONS.get(p, "None"))
            st.dataframe(disp_history, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 2: BENCHMARK EXPERIMENTS
# ---------------------------------------------------------------------------
with tab_benchmark:
    st.subheader("Controlled Benchmark Experiments")
    st.caption("Repeated trials across controlled stationary and non-stationary opponents.")

    if run_sim or "sim_results" not in st.session_state:
        with st.spinner("Running Monte Carlo experiments across random seeds..."):
            actual_agents = selected_agents if selected_agents else all_agent_names
            st.session_state.sim_results = run_experiments(
                rounds=rounds,
                seeds=seeds,
                switch_round=min(switch_round, rounds - 1),
                selected_agents=actual_agents,
            )

    results = st.session_state.sim_results
    summary = summarize(results)

    st.markdown("### Final Performance Summary")
    st.dataframe(
        summary.style.format({
            "mean_win_rate": "{:.1%}",
            "std_win_rate": "{:.1%}",
            "mean_reward": "{:.3f}",
            "mean_prediction_accuracy": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("### Performance Dynamics Over Time")

    opp_list = sorted(results["opponent"].unique())
    selected_opp = st.selectbox("Select Opponent to Analyze", opp_list, index=opp_list.index("Strategy switch") if "Strategy switch" in opp_list else 0)

    opp_df = results[results["opponent"] == selected_opp]

    # Aggregate across seeds: compute mean and standard deviation
    curve = opp_df.groupby(["agent", "round"], as_index=False).agg(
        win_rate=("win_rate", "mean"),
        win_std=("win_rate", "std"),
        pred_acc=("prediction_accuracy", "mean"),
        pred_std=("prediction_accuracy", "std"),
    )
    curve["win_std"] = curve["win_std"].fillna(0)
    curve["win_min"] = (curve["win_rate"] - curve["win_std"]).clip(0, 1)
    curve["win_max"] = (curve["win_rate"] + curve["win_std"]).clip(0, 1)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 📈 Cumulative Win Rate (Mean ± 1σ)")
        base = alt.Chart(curve).encode(x=alt.X("round:Q", title="Round"))

        band = base.mark_area(opacity=0.15).encode(
            y=alt.Y("win_min:Q", title="Cumulative Win Rate", scale=alt.Scale(domain=[0, 1])),
            y2="win_max:Q",
            color=alt.Color("agent:N", title="Agent"),
        )
        line = base.mark_line().encode(
            y="win_rate:Q",
            color=alt.Color("agent:N", title="Agent"),
            tooltip=["agent:N", "round:Q", alt.Tooltip("win_rate:Q", format=".1%"), alt.Tooltip("win_std:Q", format=".1%")],
        )
        chart = band + line

        # Add vertical reference line for strategy switch round
        if selected_opp == "Strategy switch":
            rule_df = pd.DataFrame([{"switch": switch_round}])
            rule = alt.Chart(rule_df).mark_rule(color="#e53e3e", strokeDash=[4, 4], strokeWidth=2).encode(x="switch:Q")
            text = alt.Chart(rule_df).mark_text(
                text="Strategy Shift", align="left", dx=5, dy=-120, color="#e53e3e"
            ).encode(x="switch:Q", y=alt.value(20))
            chart = chart + rule + text

        st.altair_chart(chart.interactive(), use_container_width=True)

    with chart_col2:
        st.markdown("#### 🎯 Prediction Accuracy (Mean)")
        pred_curve = curve[curve["pred_acc"].notna()]
        pred_chart = (
            alt.Chart(pred_curve)
            .mark_line()
            .encode(
                x=alt.X("round:Q", title="Round"),
                y=alt.Y("pred_acc:Q", title="Cumulative Prediction Accuracy", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("agent:N", title="Agent"),
                tooltip=["agent:N", "round:Q", alt.Tooltip("pred_acc:Q", format=".1%")],
            )
            .interactive()
        )
        if selected_opp == "Strategy switch":
            rule_df = pd.DataFrame([{"switch": switch_round}])
            rule = alt.Chart(rule_df).mark_rule(color="#e53e3e", strokeDash=[4, 4], strokeWidth=2).encode(x="switch:Q")
            pred_chart = pred_chart + rule

        st.altair_chart(pred_chart, use_container_width=True)

    st.markdown(r"""
    > [!NOTE]
    > **Concept Drift Insight:** Look closely at the **Strategy switch** opponent at the dashed red vertical line.
    > The standard **First-order Markov** model suffers a severe slump because its old transition counts dominate its memory.
    > In contrast, the **Decayed Markov** model exponentially decays older memories ($\gamma = 0.90$), adapting to the new distribution significantly faster!
    """)

    csv_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Full Round-Level Results (CSV)",
        data=csv_data,
        file_name="rps_experiment_results.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# TAB 3: MODEL BRAIN VISUALIZER (EXPLAINABILITY)
# ---------------------------------------------------------------------------
with tab_brain:
    st.subheader("Model Brain Visualizer: Learned Transition Matrix")
    st.markdown(
        "A first-order Markov model learns an empirical conditional transition matrix "
        r"$P(M_t \mid M_{t-1})$. Here you can inspect what the models learn."
    )

    brain_source = st.radio(
        "Choose Model Instance to Inspect",
        options=["From Live Arena match", "Simulated on Cyclic Opponent", "Simulated on Biased Opponent"],
        horizontal=True,
    )

    if brain_source == "From Live Arena match":
        agent_to_inspect = st.session_state.arena_agent
        source_label = f"Live Arena Agent: {st.session_state.get('current_arena_agent_name')}"
    elif brain_source == "Simulated on Cyclic Opponent":
        agent_to_inspect = MarkovAgent(seed=42)
        cyclic_seq = ["rock", "paper", "scissors"] * 40
        for m in cyclic_seq:
            agent_to_inspect.observe(m)
        source_label = "Trained on 120 rounds of Cyclic pattern (Rock -> Paper -> Scissors)"
    else:
        agent_to_inspect = MarkovAgent(seed=42)
        rng = np.random.default_rng(42)
        biased_seq = rng.choice(["rock", "paper", "scissors"], size=150, p=[0.7, 0.2, 0.1])
        for m in biased_seq:
            agent_to_inspect.observe(m)
        source_label = "Trained on 150 rounds of Biased pattern (70% Rock)"

    st.caption(f"Active Model: **{source_label}**")

    if hasattr(agent_to_inspect, "get_transition_probabilities"):
        trans_probs = agent_to_inspect.get_transition_probabilities()

        matrix_rows = []
        for m_from in MOVES:
            for m_to in MOVES:
                matrix_rows.append({
                    "Previous Move": MOVE_ICONS[m_from],
                    "Predicted Next Move": MOVE_ICONS[m_to],
                    "Probability": trans_probs[m_from][m_to],
                })
        matrix_df = pd.DataFrame(matrix_rows)

        col_hmap, col_details = st.columns([3, 2])
        with col_hmap:
            st.markdown("#### Transition Probability Heatmap $P(M_t \\mid M_{t-1})$")
            base_h = alt.Chart(matrix_df).encode(
                x=alt.X("Predicted Next Move:N", title="Next Move", sort=None),
                y=alt.Y("Previous Move:N", title="Previous Move", sort=None),
            )
            heat = base_h.mark_rect().encode(
                color=alt.Color("Probability:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), title="Probability"),
                tooltip=["Previous Move", "Predicted Next Move", alt.Tooltip("Probability:Q", format=".1%")],
            )
            text = base_h.mark_text(fontSize=14, fontWeight="bold").encode(
                text=alt.Text("Probability:Q", format=".1%"),
                color=alt.condition(alt.datum.Probability > 0.5, alt.value("white"), alt.value("black")),
            )
            h_chart = (heat + text).properties(height=260)
            st.altair_chart(h_chart, use_container_width=True)

        with col_details:
            st.markdown("#### 🔬 Inspect Learned Transitions")
            selected_prev = st.selectbox("If the previous move was:", [MOVE_ICONS[m] for m in MOVES], index=0)
            prev_key = next(m for m, icon in MOVE_ICONS.items() if icon == selected_prev)
            row_probs = trans_probs[prev_key]

            for next_m in MOVES:
                p_val = row_probs[next_m]
                st.write(f"**{MOVE_ICONS[next_m]}:** {p_val:.1%}")
                st.progress(float(p_val))

    else:
        st.info("The selected model does not have a 1st-order transition matrix (e.g. Random or Frequency). Switch to a Markov agent to inspect transitions.")


# ---------------------------------------------------------------------------
# TAB 4: METHODOLOGY & THEORY
# ---------------------------------------------------------------------------
with tab_theory:
    st.subheader("Theoretical Foundations & Mathematical Formulation")

    st.markdown(r"""
    ### 1. Game Theoretic Nash Equilibrium
    In zero-sum Rock-Paper-Scissors with payoff matrix $M \in \{-1, 0, 1\}^{3 \times 3}$, the unique mixed-strategy **Nash Equilibrium** is:
    $$\pi^* = \left(\frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right)$$
    - **Theoretical Implication:** Against an opponent executing $\pi^*$ independently at each round, no learning agent can achieve an expected win rate exceeding $33.3\%$ or an expected reward above $0.0$.
    - **Empirical Demonstration:** The uniform *Random* opponent in our benchmark demonstrates this mathematical limit.

    ---

    ### 2. Frequency Model
    Assumes moves are independent and identically distributed (i.i.d.):
    $$\hat{P}(M_t = m) = \frac{N(m)}{\sum_{m'} N(m')}$$
    $$\text{Action}_t = \text{counter}(\arg\max_m \hat{P}(M_t = m))$$
    *Failure mode:* Fails completely against state-dependent patterns (e.g., cyclic rock $\to$ paper $\to$ scissors), averaging $33.3\%$ win rate.

    ---

    ### 3. First-Order vs. Second-Order Markov Models
    - **First-Order Markov ($k=1$):**
      $$\hat{P}(M_t = m \mid M_{t-1} = s) = \frac{N(s, m)}{\sum_{m'} N(s, m')}$$
      Successfully exploits order-1 sequential patterns like cyclic loops.
    - **Second-Order Markov ($k=2$):**
      $$\hat{P}(M_t = m \mid M_{t-1} = s_1, M_{t-2} = s_2) = \frac{N(s_2, s_1, m)}{\sum_{m'} N(s_2, s_1, m')}$$
      Conditions predictions on 2-step history $(s_{t-2}, s_{t-1})$, with hierarchical fallback to first-order and global counts.

    ---

    ### 4. Concept Drift via Exponentially Decayed Markov
    When an opponent dynamically changes strategies at round $T_{switch}$ (non-stationary environment):
    $$N_t(s, m) = \gamma \cdot N_{t-1}(s, m) + \mathbb{I}(M_{t-1} = s, M_t = m)$$
    where $\gamma \in (0, 1)$ is the memory discount factor.
    - As $t$ progresses, observations $k$ steps in the past decay by $\gamma^k$.
    - For $\gamma = 0.90$, weights decay by $>95\%$ within 30 rounds ($\approx 0.90^{30} \approx 0.042$), allowing the agent to break out of inertia and track the opponent's new strategy.
    """)
