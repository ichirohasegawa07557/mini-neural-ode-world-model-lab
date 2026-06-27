from __future__ import annotations
import numpy as np
import pandas as pd
from src.env import CoupledNonlinearWorld, CoupledWorldConfig, true_causal_matrix

def pulse_schedule(n: int, target: int, start: int, duration: int, strength: float):
    return {t: np.eye(n)[target] * strength for t in range(start, start + duration)}

def generate_passive_data(episodes: int = 10, steps: int = 100, seed: int = 1) -> pd.DataFrame:
    world = CoupledNonlinearWorld(CoupledWorldConfig(seed=seed))
    return pd.concat([world.simulate(steps=steps, episode=e, label="passive") for e in range(episodes)], ignore_index=True)

def generate_interventional_data(episodes_per_target: int = 2, steps: int = 100, seed: int = 2, strength: float = 1.4) -> pd.DataFrame:
    world = CoupledNonlinearWorld(CoupledWorldConfig(seed=seed))
    frames, ep = [], 0
    for target in range(world.n):
        for k in range(episodes_per_target):
            sign = 1 if k % 2 == 0 else -1
            schedule = pulse_schedule(world.n, target, start=20, duration=30, strength=sign * strength)
            frames.append(world.simulate(steps=steps, intervention_schedule=schedule, episode=ep, label=f"intervene_x{target}"))
            ep += 1
    return pd.concat(frames, ignore_index=True)

def true_graph_table() -> pd.DataFrame:
    A = true_causal_matrix()
    rows = []
    for target in range(A.shape[0]):
        for source in range(A.shape[1]):
            rows.append({"target": f"x{target}", "source": f"x{source}", "weight": A[target, source], "edge": int(abs(A[target, source]) > 0 and target != source)})
    return pd.DataFrame(rows)
