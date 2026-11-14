# Model Artefacts

This directory contains the artefacts used by the evaluation and demo code. The
small scikit-learn fallback models are committed to Git so the headline numbers
can be reproduced without retraining. The PyTorch checkpoint is retained only
for the smoke-test audit described in Phase 5.

| Artefact | Role | Size | SHA-256 |
|---|---|---|---|
| `fallback_mlp_regressor.joblib` | Primary evaluated model (29 features, 64×64 MLP, 100k training cells, 17 epochs) | 81,704 B | `0CE407E06EF0D28D02EE44EAB65CC485A23170FECAD9454EED6C902072C5B6E5` |
| `fallback_mlp_classifier.joblib` | Companion 10-bin distance classifier used for AUC / F1 | 89,415 B | `2C20CD23A513007F3C89C1E300A5956C2414E0F90551C1F95C3C6DC267201B32` |
| `neural_heuristic_seed42.pt` | PyTorch smoke checkpoint (1 epoch, 20k cells) — **not** a trained result | 437,189 B | `F17AAFE5F9FD57AEE49F788CDE9A9F24A4C259650D3AD14330E0ABB664668A61` |

## Reproduction

```powershell
# Recreate the fallback regressor and classifier (seed 42)
python scripts/train_fallback_models.py --samples 100000 --max-iter 50 --seed 42

# Re-run the authoritative benchmark
python scripts/run_evaluation.py --backend sklearn --split test --output results/evaluation
```

The PyTorch checkpoint can be audited, but `scripts/run_torch_search.py` refuses
to benchmark it unless `--allow-untrained` is passed:

```powershell
python scripts/run_torch_search.py
python scripts/run_torch_search.py --allow-untrained --mazes 48
```

The resulting audit is written to
`results/reproducibility/torch_checkpoint_audit.json`.

## Hugging Face mirror

The project policy asks for weights to be mirrored on the Hugging Face Hub. The
fallback artefacts are small enough to version in Git and are the authoritative
copies for this deliverable; any Hub mirror must match the checksums above.
