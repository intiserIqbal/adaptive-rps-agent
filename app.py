"""Streamlit demo dashboard."""
import matplotlib.pyplot as plt
import streamlit as st

from adaptive_rps.experiment import run_experiments, summarize

st.set_page_config(page_title="Adaptive RPS Agent", page_icon="✊", layout="wide")
st.title("Adaptive Rock-Paper-Scissors Agent")
st.caption("A reproducible comparison of opponent models. Results are simulated, not proof of universal superiority.")

with st.sidebar:
    st.header("Experiment settings")
    rounds = st.slider("Rounds per match", min_value=50, max_value=1000, value=300, step=50)
    seeds = st.slider("Independent seeds", min_value=1, max_value=20, value=5)
    switch_round = st.slider("Strategy switch round", min_value=10, max_value=900, value=min(150, rounds - 1), step=10)
    run = st.button("Run experiment", type="primary")

if run or "results" not in st.session_state:
    with st.spinner("Running controlled experiments..."):
        st.session_state.results = run_experiments(rounds, seeds, min(switch_round, rounds - 1))
results = st.session_state.results
summary = summarize(results)

st.subheader("Final performance by opponent")
st.dataframe(summary.style.format({"mean_win_rate": "{:.1%}", "mean_reward": "{:.3f}", "mean_prediction_accuracy": "{:.1%}"}), use_container_width=True)

opponent = st.selectbox("Opponent", sorted(results["opponent"].unique()))
selected = results[results["opponent"] == opponent]
mean_curve = selected.groupby(["agent", "round"], as_index=False).agg(
    win_rate=("win_rate", "mean"), prediction_accuracy=("prediction_accuracy", "mean"))
left, right = st.columns(2)
with left:
    st.markdown("#### Cumulative win rate")
    fig, ax = plt.subplots()
    for agent_name, group in mean_curve.groupby("agent"):
        ax.plot(group["round"], group["win_rate"], label=agent_name)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative win rate")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("#### Prediction accuracy")
    fig, ax = plt.subplots()
    for agent_name, group in mean_curve.groupby("agent"):
        ax.plot(group["round"], group["prediction_accuracy"], label=agent_name)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative prediction accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

st.info("Prediction accuracy is calculated only on rounds where a model made a prediction. The random baseline does not predict, so its accuracy is undefined. Against a truly random opponent, historical models should not reliably outperform chance over many trials.")

csv = results.to_csv(index=False).encode("utf-8")
st.download_button("Download all round-level results (CSV)", data=csv, file_name="rps_experiment_results.csv", mime="text/csv")
