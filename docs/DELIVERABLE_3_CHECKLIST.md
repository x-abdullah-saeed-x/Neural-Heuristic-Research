# D3 research checklist (based on the newer Deliverable 3 branch)

## Stage A — Safe start

- [ ] Abort the unfinished merge and switch to `main`.
- [ ] Create `d3/rebuild-from-corrected-d2` from current `main`.
- [ ] Extract this ZIP and check `git status`: only *added* files should appear.
- [ ] Verify original D2 training reports, scripts, evaluation files and README remain unchanged.
- [ ] Record exact base commit hash with `git rev-parse HEAD`.

## Stage B — Confirm reproducibility inputs

- [ ] Confirm `data/processed/{train,val,test}.npz` and `data/cleaned/*.npz` are on your PC.
- [ ] Confirm 29-feature input order and held-out maze split.
- [ ] Retrieve/check the real trained D2 PyTorch checkpoints (seeds 42/43/44) from Hugging Face and verify hashes.
- [ ] Confirm which scikit-learn fallback model/seed/sample budget generated the D2 headline search results; avoid mixing with a differently retrained model.

## Stage C — Phase 5 experiment sequence

- [ ] RQ1: Pair each heuristic's **same** test mazes against Manhattan; report expanded nodes, runtime, optimality and confidence intervals.
- [ ] RQ1: Analyse by maze generator, dimensions and density, not just aggregate means.
- [ ] RQ2: Run features ablations with matched training budgets and seeds. Compare predictive quality **and** search costs.
- [ ] RQ3: Compare prediction fields to ground-truth h*, adjacent-cell consistency, overestimation, and representative success/failure cases.
- [ ] RQ4: Audit actual checkpoints and reproduce several search runs from the released weights.
- [ ] Select final sample counts and run all experiments again. Small test runs are for debugging only.

## Stage D — Visuals, deliverables, final report

- [ ] Build figures from **new D3 outputs**, never simply copy old D3 PNGs with different model weights.
- [ ] Generate D3-only LaTeX assets.
- [ ] Update `report/main.tex` later, manually and carefully, only after valid numbers exist.
- [ ] Retain corrected D2 methodology, training curves, results, and tables with their provenance.
- [ ] Include limitations, negative results, uncertainty, and correct contribution credits.
- [ ] Run complete repository unit tests and compile/check the final PDF.
- [ ] Open PR from the rebuilt D3 branch; review it before merging into main.

## Research interpretation safeguards

- Difference in regression MAE does **not** automatically imply improvement in A* search.
- Search quality should report both node expansions and path optimality. Runtime should include model setup/inference where appropriate.
- Admissibility and consistency are distinct; compare h(n) against h*(n) and adjacent-cell differences separately.
- The D2 multiple-seed PyTorch training results and D2 scikit-learn A* benchmark are different experiment streams. Label both correctly.
- Reproducibility claims require the specific model, seed, code/config, split and artifact hash.
