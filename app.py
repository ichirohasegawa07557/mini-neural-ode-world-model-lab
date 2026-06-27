from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Causal Neural ODE World Model Lab", layout="wide")
st.title("Causal Neural ODE World Model Lab")
st.write("Sparse causal Neural ODEs, intervention-based world-model agents, and a mini automated discovery loop.")

results = Path("results")
if not results.exists():
    st.warning("Run `python scripts/run_all.py` first.")
    st.stop()

images = [
    "true_causal_graph.png",
    "dense_learned_graph.png",
    "sparse_learned_graph.png",
    "model_comparison.png",
    "rollout_prediction.png",
    "agent_learning_curve.png",
    "agent_graph_f1.png",
    "passive_vs_active_graphs.png",
    "scientist_loop_error.png",
    "scientist_final_graph.png",
]
cols = st.columns(2)
for idx, image in enumerate(images):
    p = results / image
    if p.exists():
        with cols[idx % 2]:
            st.subheader(image)
            st.image(str(p), use_container_width=True)

gif = results / "graph_learning.gif"
if gif.exists():
    st.header("Graph learning animation")
    st.image(str(gif), use_container_width=True)

for csv in ["model_comparison.csv", "agent_comparison.csv", "intervention_history.csv", "scientist_loop_history.csv"]:
    p = results / csv
    if p.exists():
        st.subheader(csv)
        st.dataframe(pd.read_csv(p).head(30), use_container_width=True)
