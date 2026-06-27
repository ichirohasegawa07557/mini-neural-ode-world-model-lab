import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from pathlib import Path
from src.agents import run_scientist_loop
from src.visualize import plot_scientist_loop, plot_matrix, make_graph_gif

def main():
    results = Path("results"); results.mkdir(exist_ok=True)
    history, graphs, true_graph, final_binary, final_scores = run_scientist_loop(rounds=10)
    history.to_csv(results / "scientist_loop_history.csv", index=False)
    plot_scientist_loop(history, results / "scientist_loop_error.png")
    plot_matrix(final_scores, results / "scientist_final_graph.png", "Final learned graph scores")
    make_graph_gif(graphs, results / "graph_learning.gif")
    print(history.tail())

if __name__ == "__main__":
    main()
