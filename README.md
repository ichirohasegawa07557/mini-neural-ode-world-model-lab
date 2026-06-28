# Causal Neural ODE World Model Lab

A GitHub-ready research prototype that turns the ideas of causal discovery, Neural ODEs, interventions, and active world-model learning into one executable project.

## Implemented directions

### A. Causal Neural ODE

A Neural-ODE-style right-hand-side model learns:

```text
dx/dt = f_theta(x, u)
```

where `x` is the world state and `u` is an intervention vector.  
The model learns a dense version and a sparse causal version, then compares the learned causal graph with the ground-truth graph.

### B. Intervention-based world-model agent

The project compares:

```text
passive observer agent
active intervention agent
```

The passive agent only observes.  
The active agent perturbs variables and learns a world model from intervention outcomes.

### C. Mini AI Scientist loop

A closed loop is implemented:

```text
observe → train model → choose intervention → collect data → update model → repeat
```

The loop visualizes prediction error, graph recovery, intervention history, and graph-learning animation.

## Main outputs

```text
data/passive_data.csv
data/interventional_data.csv
data/true_causal_graph.csv

results/true_causal_graph.png
results/dense_learned_graph.png
results/sparse_learned_graph.png
results/model_comparison.csv
results/model_comparison.png
results/rollout_prediction.png

results/agent_comparison.csv
results/agent_learning_curve.png
results/agent_graph_f1.png
results/passive_vs_active_graphs.png
results/intervention_history.csv

results/scientist_loop_history.csv
results/scientist_loop_error.png
results/scientist_final_graph.png
results/graph_learning.gif
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest -q
```

Expected:

```text
9 passed
```

## Run all experiments

```bash
python scripts/run_all.py
```

## Run step by step

```bash
python scripts/01_generate_data.py
python scripts/02_train_causal_neural_ode.py
python scripts/03_compare_agents.py
python scripts/04_run_scientist_loop.py
streamlit run app.py
```