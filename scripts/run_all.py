import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


import subprocess, sys
def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
def main():
    py = sys.executable
    run([py, "scripts/01_generate_data.py"])
    run([py, "scripts/02_train_causal_neural_ode.py"])
    run([py, "scripts/03_compare_agents.py"])
    run([py, "scripts/04_run_scientist_loop.py"])
    print("Done. Results saved in results/.")
if __name__ == "__main__":
    main()
