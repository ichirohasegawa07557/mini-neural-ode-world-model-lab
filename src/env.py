from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

def true_causal_matrix(n: int = 5) -> np.ndarray:
    A = np.array([
        [0.00, 0.90, 0.00, 0.00, 0.00],
        [0.00, 0.00, -0.80, 0.00, 0.00],
        [0.65, 0.00, 0.00, 0.50, 0.00],
        [0.00, 0.00, 0.00, 0.00, -0.75],
        [0.55, 0.00, 0.00, 0.00, 0.00],
    ], dtype=float)
    return A[:n, :n]

@dataclass
class CoupledWorldConfig:
    n: int = 5
    dt: float = 0.05
    process_noise: float = 0.005
    seed: int = 0

class CoupledNonlinearWorld:
    def __init__(self, config: CoupledWorldConfig | None = None, A: np.ndarray | None = None):
        self.config = config or CoupledWorldConfig()
        self.n = self.config.n
        self.A = A if A is not None else true_causal_matrix(self.n)
        self.rng = np.random.default_rng(self.config.seed)
        self.damping = np.linspace(0.25, 0.45, self.n)
        self.bias = np.linspace(-0.10, 0.10, self.n)

    def rhs(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        u = np.zeros(self.n) if u is None else u
        return -self.damping * x + 0.25 * np.sin(x) + self.bias + self.A @ np.tanh(x) + u

    def step(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        noise = self.rng.normal(0, self.config.process_noise, size=self.n)
        return x + self.config.dt * self.rhs(x, u) + noise

    def simulate(self, steps: int = 120, x0=None, intervention_schedule=None, episode: int = 0, label: str = "passive") -> pd.DataFrame:
        x = self.rng.normal(0, 0.4, size=self.n) if x0 is None else x0.astype(float).copy()
        intervention_schedule = intervention_schedule or {}
        rows = []
        for t in range(steps):
            u = intervention_schedule.get(t, np.zeros(self.n))
            dx = self.rhs(x, u)
            row = {"episode": episode, "step": t, "time": t * self.config.dt, "label": label}
            for i in range(self.n):
                row[f"x{i}"] = x[i]
                row[f"dx{i}"] = dx[i]
                row[f"u{i}"] = u[i]
            rows.append(row)
            x = self.step(x, u)
        return pd.DataFrame(rows)

def graph_binary(A: np.ndarray, threshold: float = 1e-9) -> np.ndarray:
    G = (np.abs(A) > threshold).astype(int)
    np.fill_diagonal(G, 0)
    return G
