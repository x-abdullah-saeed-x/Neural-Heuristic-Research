# Model weights

- Hugging Face: https://huggingface.co/Hamid191/Neural-Hueristic-Research
- Files: neural_heuristic_seed42.pt, neural_heuristic_seed43.pt, neural_heuristic_seed44.pt
- Checkpoint SHA256: see the table below (recompute with `Get-FileHash <file> -Algorithm SHA256`)
- Load: `torch.load(path, map_location="cpu")`
- Contents: `model_state`, `model_config`, `target_mean`, `target_std`, `bin_edges` (plus `config`, `git_commit`, `seed`)
- Training settings of every run: `results/training/final_summary.csv`

| File | Seed | Size (bytes) | SHA256 | Best val MAE | Best epoch |
|---|---|---|---|---|---|
| neural_heuristic_seed42.pt | 42 | 437189 | `86565d7891f40b813879e1fe0da3c23185ae2b1dc2fa2b0e624690b2b94d6da4` | 0.2886 | 1 |
| neural_heuristic_seed43.pt | 43 | 437189 | `eb6bd9b98694451345dc6bffb1abdb45906a3ac71c8e887f345592086c6b1979` | - | - |
| neural_heuristic_seed44.pt | 44 | 437189 | `7296ca5ed751e7ee746569e8b0f9469a525a7b12624e5d7c562c1f989d219223` | - | - |
