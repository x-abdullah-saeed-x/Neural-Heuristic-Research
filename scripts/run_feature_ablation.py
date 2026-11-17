"""D3 feature ablation using D2 train/test splits without changing trained D2 models.

Usage: python scripts/run_feature_ablation.py --samples 30000 --mazes 24
Requires local data/processed/{train,test}.npz. Search requires data/cleaned/*.npz.
"""
from __future__ import annotations
import argparse
import json
from itertools import islice
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from src.evaluate import iter_split_mazes
from src.search import astar, ManhattanHeuristic
from src.sklearn_heuristic import SklearnNeuralHeuristic

FEATURE_VARIANTS = {
    'full_29': list(range(29)),
    'without_wall_window': list(range(4)),
    'goal_offsets_only': [2, 3],
    'without_goal_offsets': [0, 1] + list(range(4, 29)),
    'without_coordinates': list(range(2, 29)),
}


class SelectedFeatureModel:
    """Adapt any model trained on a feature subset to the D2 29-feature wrapper."""
    def __init__(self, regressor, feature_columns):
        self.regressor = regressor
        self.feature_columns = tuple(feature_columns)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 29:
            raise ValueError('Expected exactly 29 D2 features')
        return self.regressor.predict(X[:, self.feature_columns])


def sample_indices(length, requested, rng):
    if requested <= 0 or requested >= length:
        return np.arange(length)
    return rng.choice(length, size=requested, replace=False)


def run_variant(name, columns, X_train, y_train, X_test, y_test, mazes, seed,
                max_iter):
    clf = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam',
                       random_state=seed, max_iter=max_iter, early_stopping=True,
                       n_iter_no_change=5)
    clf.fit(X_train[:, columns], y_train)
    pred = clf.predict(X_test[:, columns])
    row = {
        'variant': name, 'n_features': len(columns), 'train_cells': len(X_train),
        'test_cells': len(X_test), 'epochs_run': int(clf.n_iter_),
        'mae_normalized': float(mean_absolute_error(y_test, pred)),
        'r2': float(r2_score(y_test, pred)),
        'mazes_evaluated': len(mazes),
    }
    if mazes:
        adapted = SelectedFeatureModel(clf, columns)
        expanded, ref_expanded, costs = [], [], []
        for m in mazes:
            h = SklearnNeuralHeuristic(adapted, m['grid'], m['goal'], mode='raw')
            n = astar(m['grid'], m['start'], m['goal'], h)
            ref = astar(m['grid'], m['start'], m['goal'], ManhattanHeuristic(m['goal']))
            expanded.append(n.nodes_expanded)
            ref_expanded.append(ref.nodes_expanded)
            costs.append(float(n.cost / m['opt_cost']) if m['opt_cost'] > 0 and n.found else float('nan'))
        row.update({
            'neural_mean_nodes': float(np.mean(expanded)),
            'manhattan_mean_nodes': float(np.mean(ref_expanded)),
            'mean_optimality_ratio': float(np.nanmean(costs)),
            'neural_to_manhattan_nodes_ratio': float(np.mean(np.asarray(expanded) / np.maximum(ref_expanded, 1))),
        })
    return row


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='data/processed/train.npz')
    p.add_argument('--test', default='data/processed/test.npz')
    p.add_argument('--samples', type=int, default=30000,
                   help='Training sample cells per variant; <=0 means all')
    p.add_argument('--test-samples', type=int, default=30000)
    p.add_argument('--mazes', type=int, default=24,
                   help='First N test mazes for search comparison; 0 skips search')
    p.add_argument('--cleaned-dir', default='data/cleaned')
    p.add_argument('--manifest', default='data/processed/split_manifest.json')
    p.add_argument('--max-iter', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='results/analysis/d3')
    a = p.parse_args(argv)
    if a.mazes < 0:
        p.error('--mazes must be nonnegative')
    rng = np.random.default_rng(a.seed)
    with np.load(a.train) as d:
        X, y = np.asarray(d['X'], dtype=np.float32), np.asarray(d['y'], dtype=np.float32).reshape(-1)
    with np.load(a.test) as d:
        Xtest, ytest = np.asarray(d['X'], dtype=np.float32), np.asarray(d['y'], dtype=np.float32).reshape(-1)
    if X.shape[1] != 29 or Xtest.shape[1] != 29:
        raise ValueError('Expected D2 29-dimensional feature vectors')
    train_idx = sample_indices(len(X), a.samples, rng)
    test_idx = sample_indices(len(Xtest), a.test_samples, rng)
    X, y = X[train_idx], y[train_idx]
    Xtest, ytest = Xtest[test_idx], ytest[test_idx]
    mazes = []
    if a.mazes:
        mazes = list(islice(iter_split_mazes('test', cleaned_dir=a.cleaned_dir,
                                           manifest_path=a.manifest), a.mazes))
        if not mazes:
            raise ValueError('No test mazes found; regenerate cleaned D2 data')
    rows = []
    for name, columns in FEATURE_VARIANTS.items():
        print(f'Training {name} with {len(columns)} features...', flush=True)
        rows.append(run_variant(name, columns, X, y, Xtest, ytest, mazes, a.seed, a.max_iter))
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / 'feature_ablation.csv', index=False)
    (dest / 'feature_ablation.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'Saved D3 ablation results under {dest}; original D2 results unaffected.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
