"""D3 command-line demo on a held-out D2 test maze; saves only new D3 outputs.

Examples:
  python scripts/demo.py --index 0 --backend none
  python scripts/demo.py --index 0 --backend sklearn --model models/fallback_mlp_regressor.joblib
  python scripts/demo.py --index 0 --backend torch --model models/neural_heuristic_seed42.pt
"""
from __future__ import annotations
import argparse
from itertools import islice
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import iter_split_mazes
from src.search import astar, ManhattanHeuristic, OracleHeuristic, ZeroHeuristic


def ascii_maze(grid, start, goal, path=(), max_size=60):
    """Draw a human-readable wall/grid/path preview with bounded output."""
    rows, cols = grid.shape
    if rows > max_size or cols > max_size:
        return f'[Maze {rows}x{cols} exceeds ASCII preview limit of {max_size}]'
    p = set(tuple(x) for x in path)
    result = []
    for r in range(rows):
        text = []
        for c in range(cols):
            spot = (r, c)
            ch = 'S' if spot == start else 'G' if spot == goal else (
                '#' if grid[r,c] else '*' if spot in p else '.')
            text.append(ch)
        result.append(''.join(text))
    return '\n'.join(result)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, default=0, help='Zero-based test maze index')
    p.add_argument('--backend', choices=['none','sklearn','torch'], default='none')
    p.add_argument('--model', default=None, help='Path to model weights for selected backend')
    p.add_argument('--mode', default='raw', choices=['raw','scaled','clipped_manhattan'])
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--cleaned-dir', default='data/cleaned')
    p.add_argument('--manifest', default='data/processed/split_manifest.json')
    p.add_argument('--output', default='results/demo/d3')
    a = p.parse_args(argv)
    if a.index < 0:
        p.error('--index cannot be negative')
    maze = next(islice(iter_split_mazes('test', cleaned_dir=a.cleaned_dir,
                                       manifest_path=a.manifest), a.index, None), None)
    if maze is None:
        raise ValueError(f'Test maze index {a.index} does not exist')
    grid, start, goal = maze['grid'], maze['start'], maze['goal']
    specs = {'dijkstra': ZeroHeuristic(), 'manhattan': ManhattanHeuristic(goal),
             'oracle': OracleHeuristic(maze['h_star'])}
    if a.backend != 'none':
        model_path = a.model or ('models/fallback_mlp_regressor.joblib' if a.backend == 'sklearn'
                                 else 'models/neural_heuristic_seed42.pt')
        if not Path(model_path).is_file():
            raise FileNotFoundError(f'Missing model {model_path}; retrieve/train model first')
        if a.backend == 'sklearn':
            from src.sklearn_heuristic import SklearnNeuralHeuristic
            model = SklearnNeuralHeuristic(model_path, grid, goal, alpha=a.alpha, mode=a.mode)
        else:
            from src.heuristics import NeuralHeuristic
            model = NeuralHeuristic(model_path, grid, goal, alpha=a.alpha, mode=a.mode)
        specs[f'{a.backend}_{a.mode}'] = model
    records = {}
    for name, heuristic in specs.items():
        found = astar(grid, start, goal, heuristic)
        records[name] = {'found': found.found, 'cost': found.cost,
                         'nodes_expanded': found.nodes_expanded, 'runtime_ms': found.runtime_ms}
        print(f'{name}: cost={found.cost}, nodes={found.nodes_expanded}, found={found.found}')
    result = astar(grid, start, goal, ManhattanHeuristic(goal))
    view = ascii_maze(grid, start, goal, result.path)
    print('\nManhattan A* path:\n' + view)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'demo.json').write_text(json.dumps({'maze_index': a.index,
                                                  'maze_id': str(maze['maze_id']),
                                                  'optimal_cost': maze['opt_cost'],
                                                  'results': records}, indent=2), encoding='utf-8')
    (output / 'maze_preview.txt').write_text(view + '\n', encoding='utf-8')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
