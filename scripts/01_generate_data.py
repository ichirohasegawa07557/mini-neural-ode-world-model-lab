import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from pathlib import Path
from src.data import generate_passive_data, generate_interventional_data, true_graph_table
from src.env import true_causal_matrix
from src.visualize import plot_matrix

def main():
    data, results = Path("data"), Path("results")
    data.mkdir(exist_ok=True); results.mkdir(exist_ok=True)
    generate_passive_data(episodes=10, steps=100).to_csv(data / "passive_data.csv", index=False)
    generate_interventional_data(episodes_per_target=2, steps=100).to_csv(data / "interventional_data.csv", index=False)
    true_graph_table().to_csv(data / "true_causal_graph.csv", index=False)
    plot_matrix(abs(true_causal_matrix()), results / "true_causal_graph.png", "Ground-truth causal graph")
    print("Generated data.")

if __name__ == "__main__":
    main()
