from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def savefig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_matrix(mat, path, title):
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(mat, cmap="viridis")
    plt.colorbar(label="edge score")
    n = mat.shape[0]
    plt.xticks(range(n), [f"x{i}" for i in range(n)])
    plt.yticks(range(n), [f"x{i}" for i in range(n)])
    plt.xlabel("source")
    plt.ylabel("target")
    plt.title(title)
    savefig(path)

def plot_model_comparison(df, path):
    plt.figure(figsize=(7, 4.5))
    plt.bar(df["model"], df["train_mse"])
    plt.ylabel("training derivative MSE")
    plt.title("Dense vs sparse causal Neural ODE")
    plt.xticks(rotation=15, ha="right")
    savefig(path)

def plot_rollout(true_df, pred_df, path):
    plt.figure(figsize=(8, 5))
    for c in ["x0", "x1", "x2"]:
        plt.plot(true_df["time"], true_df[c], label=f"true {c}", linewidth=1.4)
        plt.plot(pred_df["time"], pred_df[c], "--", label=f"pred {c}", linewidth=1.0)
    plt.xlabel("time")
    plt.ylabel("state")
    plt.title("Sparse causal Neural ODE rollout")
    plt.grid(True, linestyle="--", linewidth=0.4)
    plt.legend(ncol=2, fontsize=8)
    savefig(path)

def plot_agent_learning(history, path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["round"], history["passive_test_mse"], marker="o", label="passive observer")
    plt.plot(history["round"], history["active_test_mse"], marker="o", label="active intervention agent")
    plt.xlabel("round")
    plt.ylabel("test MSE")
    plt.title("World-model learning: passive vs active")
    plt.grid(True, linestyle="--", linewidth=0.4)
    plt.legend()
    savefig(path)

def plot_agent_graph_f1(history, path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["round"], history["passive_graph_f1"], marker="o", label="passive graph F1")
    plt.plot(history["round"], history["active_graph_f1"], marker="o", label="active graph F1")
    plt.xlabel("round")
    plt.ylabel("graph F1")
    plt.title("Causal graph recovery")
    plt.grid(True, linestyle="--", linewidth=0.4)
    plt.legend()
    savefig(path)

def plot_three_graphs(true_g, passive_g, active_g, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, mat, title in zip(axes, [true_g, passive_g, active_g], ["true graph", "passive learned", "active learned"]):
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("source")
        ax.set_ylabel("target")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_scientist_loop(history, path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["round"], history["test_mse"], marker="o", label="test MSE")
    plt.xlabel("round")
    plt.ylabel("test MSE")
    plt.title("Mini AI Scientist loop: model error")
    plt.grid(True, linestyle="--", linewidth=0.4)
    plt.legend()
    savefig(path)

def make_graph_gif(graphs, path):
    imgs = []
    for i, g in enumerate(graphs):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(g, cmap="viridis")
        ax.set_title(f"learned graph scores | round {i}")
        ax.set_xlabel("source")
        ax.set_ylabel("target")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, imgs, duration=0.35)
