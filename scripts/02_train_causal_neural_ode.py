import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from pathlib import Path
import numpy as np
import pandas as pd
from src.env import CoupledNonlinearWorld, CoupledWorldConfig, graph_binary
from src.data import pulse_schedule, generate_interventional_data
from src.causal_neural_ode import fit_model, graph_metrics, rollout_model
from src.visualize import plot_matrix, plot_model_comparison, plot_rollout

def main():
    data, results = Path("data"), Path("results")
    data.mkdir(exist_ok=True); results.mkdir(exist_ok=True)
    p = data / "interventional_data.csv"
    if not p.exists():
        generate_interventional_data().to_csv(p, index=False)
    df = pd.read_csv(p)
    dense = fit_model(df, sparse=False)
    sparse = fit_model(df, sparse=True, threshold=0.07)
    world = CoupledNonlinearWorld(CoupledWorldConfig(seed=99))
    true_g = graph_binary(world.A)
    cmp = pd.DataFrame([
        {"model": "dense_neural_ode", "train_mse": dense.train_mse, **graph_metrics(dense.adjacency_binary, true_g)},
        {"model": "sparse_causal_neural_ode", "train_mse": sparse.train_mse, **graph_metrics(sparse.adjacency_binary, true_g)},
    ])
    cmp.to_csv(results / "model_comparison.csv", index=False)
    plot_matrix(dense.adjacency_scores, results / "dense_learned_graph.png", "Dense Neural ODE learned graph")
    plot_matrix(sparse.adjacency_scores, results / "sparse_learned_graph.png", "Sparse causal Neural ODE learned graph")
    plot_model_comparison(cmp, results / "model_comparison.png")
    x0 = df[[f"x{i}" for i in range(5)]].iloc[0].values
    schedule = pulse_schedule(5, target=2, start=25, duration=25, strength=1.2)
    true_rollout = world.simulate(steps=100, x0=x0, intervention_schedule=schedule, label="true_test")
    pred_rollout = rollout_model(sparse, x0=x0, steps=100, dt=world.config.dt, intervention_schedule=schedule)
    true_rollout.to_csv(results / "true_rollout.csv", index=False)
    pred_rollout.to_csv(results / "sparse_model_rollout.csv", index=False)
    plot_rollout(true_rollout, pred_rollout, results / "rollout_prediction.png")
    print(cmp)

if __name__ == "__main__":
    main()
