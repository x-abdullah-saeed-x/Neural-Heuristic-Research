"""Planned ablations from the proposal (Phase 4.2).

1. Obstacle density 10/20/30/40 %  -> per-density nodes / optimality / latency
2. Model size 1 / 2 / 4 hidden layers -> break-even of inference cost vs node savings
3. Gamma in [0.5, 1.5] scaled heuristic -> optimality vs speed

Usage (repo root):
  python scripts/run_ablations.py --samples 50000 --max-iter 30
Outputs: results/ablations/{density,model_size,gamma}_ablation.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluate import iter_split_mazes, run_search_evaluation
from src.search import ManhattanHeuristic
from src.sklearn_heuristic import SklearnNeuralHeuristic

LAYERS = {"1_layer": (64,), "2_layer": (64, 64), "4_layer": (64, 64, 64, 64)}
GAMMAS = [0.5, 0.75, 1.0, 1.25, 1.5]


def summarise(df, by):
    g = df.groupby(by).agg(
        nodes=("nodes_expanded", "mean"), latency_ms=("runtime_ms", "mean"),
        search_ms=("search_ms", "mean"), setup_ms=("setup_ms", "mean"),
        optimality=("optimality_ratio", "mean"), success=("found", "mean")).reset_index()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.npz")
    ap.add_argument("--cleaned-dir", default="data/cleaned")
    ap.add_argument("--out", default="results/ablations")
    ap.add_argument("--samples", type=int, default=50000)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.train)
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(d["X"]), size=min(a.samples, len(d["X"])), replace=False)
    X, y = d["X"][idx], d["y"][idx]
    mazes = list(iter_split_mazes(split="test", cleaned_dir=a.cleaned_dir, seed=a.seed))

    models = {}
    for name, layers in LAYERS.items():
        m = MLPRegressor(hidden_layer_sizes=layers, max_iter=a.max_iter, random_state=a.seed,
                         early_stopping=True, n_iter_no_change=5)
        models[name] = m.fit(X, y)

    # (2) model size, with the clipped-Manhattan and raw variants
    heur = {"manhattan": (lambda m: ManhattanHeuristic(m["goal"]), 1.0)}
    for name, mdl in models.items():
        heur[f"{name}_raw"] = (lambda m, mdl=mdl: SklearnNeuralHeuristic(mdl, m["grid"], m["goal"], 1.0, "raw"), 1.0)
        heur[f"{name}_clipped"] = (lambda m, mdl=mdl: SklearnNeuralHeuristic(mdl, m["grid"], m["goal"], 1.0, "clipped_manhattan"), 1.0)
    df = run_search_evaluation(mazes, heur, a.out, split="model_size", seed=a.seed)
    summarise(df, "heuristic").to_csv(f"{a.out}/model_size_ablation.csv", index=False)
    # (1) density ablation from the same runs
    summarise(df, ["density", "heuristic"]).to_csv(f"{a.out}/density_ablation.csv", index=False)

    # (3) gamma sweep on the 2-layer model
    heur = {"manhattan": (lambda m: ManhattanHeuristic(m["goal"]), 1.0)}
    for g in GAMMAS:
        heur[f"gamma_{g}"] = (lambda m, g=g: SklearnNeuralHeuristic(models["2_layer"], m["grid"], m["goal"], g, "scaled"), g)
    df = run_search_evaluation(mazes, heur, a.out, split="gamma", seed=a.seed)
    summarise(df, "heuristic").to_csv(f"{a.out}/gamma_ablation.csv", index=False)
    print("Ablations written to", a.out)


if __name__ == "__main__":
    main()
