from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class FitResult:
    model_name: str
    coefficients: np.ndarray
    feature_names: list[str]
    adjacency_scores: np.ndarray
    adjacency_binary: np.ndarray
    train_mse: float

def infer_dimension(df: pd.DataFrame) -> int:
    return len([c for c in df.columns if c.startswith("x") and c[1:].isdigit()])

def build_features(df: pd.DataFrame):
    n = infer_dimension(df)
    X = df[[f"x{i}" for i in range(n)]].values
    U = df[[f"u{i}" for i in range(n)]].values
    blocks = [np.ones((len(df), 1)), X, np.tanh(X), X**2, np.sin(X), U]
    names = ["bias"] + [f"x{i}" for i in range(n)] + [f"tanh_x{i}" for i in range(n)] + [f"x{i}^2" for i in range(n)] + [f"sin_x{i}" for i in range(n)] + [f"u{i}" for i in range(n)]
    return np.concatenate(blocks, axis=1), names

def derivative_matrix(df: pd.DataFrame) -> np.ndarray:
    n = infer_dimension(df)
    return df[[f"dx{i}" for i in range(n)]].values

def ridge_fit(Theta, Y, alpha=1e-3):
    return np.linalg.solve(Theta.T @ Theta + alpha * np.eye(Theta.shape[1]), Theta.T @ Y)

def sparse_fit(Theta, Y, alpha=1e-3, threshold=0.06, iters=5):
    W = ridge_fit(Theta, Y, alpha)
    for _ in range(iters):
        W[np.abs(W) < threshold] = 0
        for k in range(Y.shape[1]):
            active = np.abs(W[:, k]) > 0
            if active.sum() == 0:
                continue
            Wa = ridge_fit(Theta[:, active], Y[:, [k]], alpha).ravel()
            W[:, k] = 0
            W[active, k] = Wa
    return W

def adjacency_from_coefficients(W, names, n, threshold=0.08):
    scores = np.zeros((n, n))
    for fi, name in enumerate(names):
        for source in range(n):
            if name in {f"x{source}", f"tanh_x{source}", f"x{source}^2", f"sin_x{source}"}:
                scores[:, source] += np.abs(W[fi, :])
    np.fill_diagonal(scores, 0)
    binary = (scores > threshold).astype(int)
    np.fill_diagonal(binary, 0)
    return scores, binary

def fit_model(df: pd.DataFrame, sparse: bool = True, threshold: float = 0.06) -> FitResult:
    n = infer_dimension(df)
    Theta, names = build_features(df)
    Y = derivative_matrix(df)
    W = sparse_fit(Theta, Y, threshold=threshold) if sparse else ridge_fit(Theta, Y)
    pred = Theta @ W
    mse = float(np.mean((pred - Y) ** 2))
    scores, binary = adjacency_from_coefficients(W, names, n)
    name = "sparse_causal_neural_ode" if sparse else "dense_neural_ode"
    return FitResult(name, W, names, scores, binary, mse)

def rhs_from_model(x, u, fit: FitResult):
    theta = np.array([1.0] + list(x) + list(np.tanh(x)) + list(x**2) + list(np.sin(x)) + list(u))
    return theta @ fit.coefficients

def rollout_model(fit: FitResult, x0, steps=100, dt=0.05, intervention_schedule=None):
    n = len(x0)
    x = x0.astype(float).copy()
    intervention_schedule = intervention_schedule or {}
    rows = []
    for t in range(steps):
        u = intervention_schedule.get(t, np.zeros(n))
        dx = rhs_from_model(x, u, fit)
        row = {"step": t, "time": t * dt}
        for i in range(n):
            row[f"x{i}"] = x[i]
            row[f"dx{i}"] = dx[i]
            row[f"u{i}"] = u[i]
        rows.append(row)
        x = x + dt * dx
    return pd.DataFrame(rows)

def graph_metrics(pred_binary, true_binary):
    mask = ~np.eye(true_binary.shape[0], dtype=bool)
    pred, true = pred_binary[mask].astype(int), true_binary[mask].astype(int)
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}
