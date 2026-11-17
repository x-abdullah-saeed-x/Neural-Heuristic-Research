"""D3 PyTorch checkpoint audit/search on the corrected D2 model (not the old smoke run).

Usage:
  python scripts/run_torch_search.py --checkpoint models/neural_heuristic_seed42.pt --audit-only
  python scripts/run_torch_search.py --checkpoint models/neural_heuristic_seed42.pt --mazes 24
All outputs are written to results/analysis/d3/ by default.
"""
from __future__ import annotations
import argparse
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.evaluate import iter_split_mazes
from src.heuristics import NeuralHeuristic
from src.search import astar, ManhattanHeuristic


def audit_checkpoint(path: str) -> dict:
    f = Path(path)
    if not f.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {f}. Retrieve your trained D2 weights from Hugging Face.')
    checkpoint = torch.load(str(f), map_location='cpu', weights_only=False)
    if not {'model_state','model_config'}.issubset(checkpoint):
        raise ValueError('This is not a compatible D2 PyTorch checkpoint')
    config = checkpoint.get('config', {})
    train_cfg = config.get('train', {}) if isinstance(config, dict) else {}
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    info = {
        'checkpoint_name': f.name,
        'sha256': sha,
        'seed': checkpoint.get('seed'),
        'git_commit': checkpoint.get('git_commit'),
        'best_epoch': checkpoint.get('best_epoch'),
        'val_mae_normalized': checkpoint.get('val_mae'),
        'configured_epochs': train_cfg.get('epochs'),
        'training_cells_per_epoch': train_cfg.get('max_train_cells_per_epoch'),
        'architecture': checkpoint.get('model_config'),
        'model_parameter_count': int(sum(x.numel() for x in checkpoint['model_state'].values())),
        'note': 'Report these checkpoint metadata as observed, not as a smoke-test assumption.'
    }
    return info


def benchmark(checkpoint: str, mazes: list[dict], mode: str, alpha: float) -> dict:
    rows = []
    for i, m in enumerate(mazes):
        neural = NeuralHeuristic(checkpoint, m['grid'], m['goal'], alpha=alpha, mode=mode)
        n = astar(m['grid'], m['start'], m['goal'], neural)
        baseline = astar(m['grid'], m['start'], m['goal'], ManhattanHeuristic(m['goal']))
        rows.append({
            'maze_id': str(m['maze_id']), 'neural_nodes': n.nodes_expanded,
            'manhattan_nodes': baseline.nodes_expanded,
            'neural_cost': n.cost, 'optimal_cost': m['opt_cost'],
            'optimality_ratio': float(n.cost / m['opt_cost']) if n.found and m['opt_cost'] > 0 else None,
        })
        if (i + 1) % 10 == 0:
            print(f'Benchmarked {i + 1}/{len(mazes)} mazes', flush=True)
    return {
        'backend': 'torch', 'checkpoint_name': Path(checkpoint).name,
        'mode': mode, 'alpha': alpha, 'n_mazes': len(rows),
        'manhattan_mean_nodes': float(np.mean([r['manhattan_nodes'] for r in rows])),
        'neural_mean_nodes': float(np.mean([r['neural_nodes'] for r in rows])),
        'per_maze': rows,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/neural_heuristic_seed42.pt')
    p.add_argument('--mazes', type=int, default=24)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--mode', choices=['raw', 'scaled', 'clipped_manhattan'], default='raw')
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--cleaned-dir', default='data/cleaned')
    p.add_argument('--manifest', default='data/processed/split_manifest.json')
    p.add_argument('--output', default='results/analysis/d3')
    a = p.parse_args(argv)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    audit = audit_checkpoint(a.checkpoint)
    (out / 'torch_checkpoint_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    if a.audit_only:
        return 0
    if a.mazes <= 0:
        p.error('--mazes must be positive unless --audit-only is set')
    mazes = list(islice(iter_split_mazes('test', cleaned_dir=a.cleaned_dir,
                                       manifest_path=a.manifest), a.mazes))
    if not mazes:
        raise ValueError('No test mazes found. Recreate the cleaned D2 dataset first.')
    result = benchmark(a.checkpoint, mazes, a.mode, a.alpha)
    (out / 'torch_search.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Torch {result['neural_mean_nodes']:.2f} vs Manhattan {result['manhattan_mean_nodes']:.2f} nodes")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
