"""Create a full Deliverable 1--3 LaTeX report WITHOUT editing the D2 source.

Run from anywhere:
    python report/d3_final/build_report.py

The output is report/d3_final/main.tex, derived from the local report/main.tex
on your corrected D2 base. The phase-5 body is supplied by phase5_results.tex.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'report' / 'main.tex'
OUTPUT = ROOT / 'report' / 'd3_final' / 'main.tex'
PHASE5 = ROOT / 'report' / 'd3_final' / 'phase5_results.tex'

ABSTRACT = r"""\begin{abstract}
This consolidated report compares Manhattan A$^\ast$ with learned neural
heuristics on synthetic, static grid mazes. Deliverables 1 and 2 establish
maze generation, ground-truth labels, model design, training, and a separate
scikit-learn fallback benchmark. Deliverable 3 evaluates the corrected,
fully trained PyTorch checkpoint (seed 42; validation MAE $0.05161$ in
$h^\ast/N$ units) on 282 held-out mazes. Neural A$^\ast$ expands
498.58 nodes per maze versus 180.79 for Manhattan, a mean paired difference
of $+317.78$ nodes (95\% bootstrap CI $[256.52,385.52]$; Wilcoxon
$p=5.81\times10^{-43}$). The neural algorithm finds optimal paths on 232
mazes and has a mean optimality ratio of 1.0082. Field diagnostics identify
55.32\% overestimated reachable-cell values and 48.44\% locally inconsistent
neighbor pairs. Independent feature ablations show that goal offsets are
important for search, but no measured learned variant reduces expansions
below Manhattan on average. The proposed search-efficiency hypothesis is
not supported by the observed experiments.
\end{abstract}"""

CONCLUSION = r"""\section{Conclusion}
The completed project separates three levels of evidence. Deliverable 1
establishes labelled maze data and deterministic splits. Deliverable 2
implements a multitask PyTorch network, records multi-seed training, and
reports a separate scikit-learn fallback evaluation. Deliverable 3 evaluates
the trained seed-42 PyTorch checkpoint directly on all 282 held-out mazes.
Although average path quality is near optimal (mean ratio 1.0082), the raw
neural A$^\ast$ heuristic expands 498.58 nodes per maze compared with
180.79 for Manhattan. The paired difference and subgroup diagnostics
support a negative conclusion for the hypothesized expansion savings.
Many reachable cells exhibit overestimation or inconsistency, but the
observed statistical relationships are not sufficient to prove a single
causal explanation. Further work should compare multiple trained seeds,
search-aware objectives, and larger or dynamic environments. These
conclusions use the new PyTorch search experiment; the earlier fallback
results remain separately documented in Phase 4.
"""


def replace_once(text: str, old: str, new: str, description: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f'Expected exactly one {description} anchor; found {text.count(old)}')
    return text.replace(old, new, 1)


def build(source: str) -> str:
    # Retain the entire corrected D2 document, changing only D3-specific
    # metadata and adding the new D3 phase.
    source = replace_once(
        source,
        r'\large Deliverable 2 --- Model Architecture \& Empirical Evaluation}',
        r'\large Deliverable 3 --- Consolidated Research Report (Phases 1--5)}',
        'title',
    )
    source = replace_once(
        source,
        r'Neural-Guided Heuristics versus Classical Pathfinding\\ in Dynamic Grid-Mazes',
        r'Neural-Guided Heuristics versus Classical Pathfinding\\ in Procedural Grid Mazes',
        'title topic',
    )
    pattern = r'\\begin\{abstract\}.*?\\end\{abstract\}'
    source, count = re.subn(pattern, lambda _: ABSTRACT, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError('Could not locate original D2 abstract')
    source = replace_once(
        source,
        r'\input{report/sections/phase4_evaluation.tex}',
        r'\input{report/sections/phase4_evaluation.tex}'
        '\n' + r'\input{report/d3_final/phase5_results.tex}',
        'phase-4 input',
    )
    # D2 phase3_training.tex contains \input{generated/*.tex}. Keeping a search
    # path avoids editing the D2 file or its generated tables.
    search_path = ('\\makeatletter\n'
                   '\\def\\input@path{{report/}{report/generated/}{report/sections/}}\n'
                   '\\makeatother\n')
    source = replace_once(source, r'\graphicspath{{figures/}}',
                          r'\graphicspath{{figures/}{report/}}' + '\n' + search_path,
                          'figure/search path')
    conclusion_pattern = (
        r'\\section\{Conclusion\}.*?'
        r'(?=\n% ={10,}\n\\begin\{thebibliography\})'
    )
    source, count = re.subn(conclusion_pattern, lambda _: CONCLUSION, source,
                            count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError('Could not uniquely locate D2 conclusion')
    # Make paths in generated document explicit and include source provenance.
    source = ('% GENERATED from report/main.tex using report/d3_final/build_report.py\n'
              '% Original D2 report/main.tex was NOT changed.\n'
              '% Compile from the repository root.\n' + source)
    return source


def main():
    if not ORIGINAL.is_file():
        raise SystemExit(f'Missing original D2 report: {ORIGINAL}')
    if not PHASE5.is_file():
        raise SystemExit(f'Missing D3 section: {PHASE5}')
    original = ORIGINAL.read_text(encoding='utf-8-sig')
    combined = build(original)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(combined, encoding='utf-8')
    print(f'Created {OUTPUT.relative_to(ROOT)}')
    print(f'Original remains unchanged: {ORIGINAL.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
