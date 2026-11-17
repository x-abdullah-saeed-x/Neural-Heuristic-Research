"""Make fresh D3 prediction-field figures from corrected D2 checkpoints.

Usage: python scripts/sample_predictions.py --backend torch --model models/neural_heuristic_seed42.pt
Output: figures/d3/prediction_fields.png
"""
from __future__ import annotations
import argparse
from itertools import islice
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.evaluate import iter_split_mazes


def make_model(backend, model_path, m):
    if backend == 'sklearn':
        from src.sklearn_heuristic import SklearnNeuralHeuristic
        obj = SklearnNeuralHeuristic(model_path, m['grid'], m['goal'])
        return np.asarray(obj._pred_grid)
    from src.heuristics import NeuralHeuristic
    return NeuralHeuristic(model_path, m['grid'], m['goal']).predictions()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['sklearn', 'torch'], default='torch')
    p.add_argument('--model', default=None)
    p.add_argument('--mazes', type=int, default=3)
    p.add_argument('--output', default='figures/d3/prediction_fields.png')
    p.add_argument('--cleaned-dir', default='data/cleaned')
    p.add_argument('--manifest', default='data/processed/split_manifest.json')
    a = p.parse_args(argv)
    model = a.model or ('models/neural_heuristic_seed42.pt' if a.backend == 'torch' else
                        'models/fallback_mlp_regressor.joblib')
    if not Path(model).is_file():
        raise FileNotFoundError(f'{model} missing. Retrieve or regenerate model first.')
    if a.mazes < 1:
        p.error('--mazes must be positive')
    mazes = list(islice(iter_split_mazes('test', cleaned_dir=a.cleaned_dir,
                                       manifest_path=a.manifest), a.mazes))
    if not mazes:
        raise ValueError('No test mazes available')
    fig, axes = plt.subplots(len(mazes), 3, figsize=(11, 3.4 * len(mazes)), squeeze=False)
    for i, m in enumerate(mazes):
        grid, true = np.asarray(m['grid']), np.asarray(m['h_star'], dtype=float)
        pred = make_model(a.backend, model, m)
        mask = (grid == 0) & np.isfinite(true) & (true >= 0)
        truth_map = np.where(mask, true, np.nan)
        pred_map = np.where(mask, pred, np.nan)
        diff_map = np.where(mask, pred - true, np.nan)
        for ax, vals, title in zip(axes[i], (truth_map, pred_map, diff_map),
                                   ('True cost h*', 'Predicted cost', 'Error (prediction - truth)')):
            ax.imshow(vals)
            ax.set_title(f'Maze {i+1}: {title}')
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f'Saved new D3 figure: {output}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
