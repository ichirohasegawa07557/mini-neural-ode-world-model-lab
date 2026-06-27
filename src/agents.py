from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.env import CoupledNonlinearWorld, CoupledWorldConfig, graph_binary
from src.data import pulse_schedule
from src.causal_neural_ode import fit_model, graph_metrics, build_features, derivative_matrix

@dataclass
class AgentExperimentResult:
    history: pd.DataFrame
    passive_graph: np.ndarray
    active_graph: np.ndarray
    true_graph: np.ndarray
    intervention_history: pd.DataFrame

def evaluate_model(fit, test_df):
    Theta, _ = build_features(test_df)
    Y = derivative_matrix(test_df)
    return float(np.mean((Theta @ fit.coefficients - Y) ** 2))

def collect_passive(world, episodes=1, steps=100):
    return pd.concat([world.simulate(steps=steps, episode=e, label="passive_agent") for e in range(episodes)], ignore_index=True)

def collect_intervention(world, target, episode, steps=100, strength=1.4):
    sign = 1 if episode % 2 == 0 else -1
    schedule = pulse_schedule(world.n, target, 20, 30, sign * strength)
    return world.simulate(steps=steps, intervention_schedule=schedule, episode=episode, label=f"active_x{target}")

def run_agent_comparison(rounds=8, seed=10):
    pw = CoupledNonlinearWorld(CoupledWorldConfig(seed=seed))
    aw = CoupledNonlinearWorld(CoupledWorldConfig(seed=seed+1))
    tw = CoupledNonlinearWorld(CoupledWorldConfig(seed=seed+2))
    true_g = graph_binary(pw.A)
    test_df = collect_passive(tw, episodes=3, steps=100)
    passive_data, active_data, history, interventions = [], [], [], []
    target = 0
    for r in range(rounds):
        passive_data.append(collect_passive(pw, 1, 100))
        active_data.append(collect_intervention(aw, target, r, 100))
        interventions.append({"round": r, "target": f"x{target}"})
        pf = fit_model(pd.concat(passive_data, ignore_index=True), sparse=True, threshold=0.07)
        af = fit_model(pd.concat(active_data, ignore_index=True), sparse=True, threshold=0.07)
        pg = graph_metrics(pf.adjacency_binary, true_g)
        ag = graph_metrics(af.adjacency_binary, true_g)
        history.append({"round": r, "passive_test_mse": evaluate_model(pf, test_df), "active_test_mse": evaluate_model(af, test_df), "passive_graph_f1": pg["f1"], "active_graph_f1": ag["f1"], "active_target": target})
        confidence = af.adjacency_scores.sum(axis=0) + af.adjacency_scores.sum(axis=1)
        target = int(np.argmin(confidence + 0.01 * np.arange(aw.n)))
    return AgentExperimentResult(pd.DataFrame(history), pf.adjacency_binary, af.adjacency_binary, true_g, pd.DataFrame(interventions))

def run_scientist_loop(rounds=10, seed=30):
    world = CoupledNonlinearWorld(CoupledWorldConfig(seed=seed))
    test_world = CoupledNonlinearWorld(CoupledWorldConfig(seed=seed+1))
    true_g = graph_binary(world.A)
    test_df = collect_passive(test_world, 3, 100)
    data, history, graphs, target = [], [], [], 0
    for r in range(rounds):
        data.append(collect_intervention(world, target, r, 100))
        fit = fit_model(pd.concat(data, ignore_index=True), sparse=True, threshold=0.07)
        gm = graph_metrics(fit.adjacency_binary, true_g)
        history.append({"round": r, "intervention_target": f"x{target}", "test_mse": evaluate_model(fit, test_df), "graph_f1": gm["f1"], "precision": gm["precision"], "recall": gm["recall"]})
        graphs.append(fit.adjacency_scores.copy())
        confidence = fit.adjacency_scores.sum(axis=0) + fit.adjacency_scores.sum(axis=1)
        target = int(np.argmin(confidence + 0.01 * np.arange(world.n)))
    return pd.DataFrame(history), graphs, true_g, fit.adjacency_binary, fit.adjacency_scores
