import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from pathlib import Path
from src.agents import run_agent_comparison
from src.visualize import plot_agent_learning, plot_agent_graph_f1, plot_three_graphs

def main():
    results = Path("results"); results.mkdir(exist_ok=True)
    res = run_agent_comparison(rounds=8)
    res.history.to_csv(results / "agent_comparison.csv", index=False)
    res.intervention_history.to_csv(results / "intervention_history.csv", index=False)
    plot_agent_learning(res.history, results / "agent_learning_curve.png")
    plot_agent_graph_f1(res.history, results / "agent_graph_f1.png")
    plot_three_graphs(res.true_graph, res.passive_graph, res.active_graph, results / "passive_vs_active_graphs.png")
    print(res.history.tail())

if __name__ == "__main__":
    main()
