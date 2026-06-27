import numpy as np
from src.env import CoupledNonlinearWorld, CoupledWorldConfig, true_causal_matrix, graph_binary
from src.data import generate_passive_data, generate_interventional_data
from src.causal_neural_ode import fit_model, rollout_model, graph_metrics
from src.agents import run_agent_comparison, run_scientist_loop

def test_true_graph_shape():
    assert true_causal_matrix().shape == (5, 5)

def test_world_simulates():
    df = CoupledNonlinearWorld(CoupledWorldConfig(seed=0)).simulate(steps=20)
    assert len(df) == 20 and "x0" in df.columns

def test_passive_data_generation():
    assert len(generate_passive_data(episodes=2, steps=20)) == 40

def test_interventional_data_generation():
    df = generate_interventional_data(episodes_per_target=1, steps=20)
    assert len(df) == 100 and any(df["label"].str.contains("intervene"))

def test_fit_model_runs():
    df = generate_interventional_data(episodes_per_target=1, steps=30)
    fit = fit_model(df, sparse=True)
    assert fit.coefficients.shape[1] == 5
    assert fit.adjacency_binary.shape == (5, 5)

def test_rollout_runs():
    df = generate_interventional_data(episodes_per_target=1, steps=30)
    fit = fit_model(df, sparse=True)
    x0 = df[[f"x{i}" for i in range(5)]].iloc[0].values
    out = rollout_model(fit, x0, steps=10)
    assert len(out) == 10

def test_graph_metrics_runs():
    A = graph_binary(true_causal_matrix())
    assert graph_metrics(A, A)["f1"] > 0.99

def test_agents_run():
    assert len(run_agent_comparison(rounds=2).history) == 2

def test_scientist_loop_runs():
    hist, graphs, *_ = run_scientist_loop(rounds=2)
    assert len(hist) == 2 and len(graphs) == 2
