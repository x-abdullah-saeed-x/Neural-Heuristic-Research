import numpy as np
from scripts.demo import ascii_maze
from scripts.make_phase5_report_assets import esc


def test_ascii_path_and_walls():
    grid = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    preview = ascii_maze(grid, (0, 0), (2, 2), [(0,0),(1,0),(1,1),(2,1),(2,2)])
    assert preview.splitlines() == ['S.#', '**.', '#*G']


def test_ascii_large_grid_limits_output():
    grid = np.zeros((100, 100), dtype=int)
    assert 'exceeds ASCII preview limit' in ascii_maze(grid, (0,0), (99,99))


def test_latex_escaping():
    assert esc('goal_offsets') == r'goal\_offsets'
