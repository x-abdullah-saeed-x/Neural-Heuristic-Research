"""Upload the trained model files to the Hugging Face Hub.

Run from the repository root AFTER training and after
`python scripts/write_models_readme.py --hf-user <user> --repo <repo>`.

    set HF_TOKEN=hf_xxx            (Windows cmd; PowerShell: $env:HF_TOKEN="hf_xxx")
    python scripts/upload_to_hf.py --repo-id Hamid191/Neural-Hueristic-Research

The token needs WRITE access to that repo (or be logged in with `hf auth login`).
Use --dry-run to list what would be uploaded without sending anything.
Use --create-pr if the repo belongs to someone else (the owner then merges the pull request).

Uploaded: models/neural_heuristic_seed*.pt, models/fallback_mlp_*.joblib,
and models/README.md (shown as the model card, README.md, on the Hub).
"""
import argparse
import glob
import os
import sys


def collect(models_dir):
    files = sorted(glob.glob(os.path.join(models_dir, "neural_heuristic_seed*.pt")))
    files += sorted(glob.glob(os.path.join(models_dir, "fallback_mlp_*.joblib")))
    pairs = [(f, os.path.basename(f)) for f in files]
    readme = os.path.join(models_dir, "README.md")
    if os.path.exists(readme):
        pairs.append((readme, "README.md"))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Upload model files to Hugging Face")
    ap.add_argument("--repo-id", required=True, help="e.g. Hamid191/Neural-Hueristic-Research")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--create-pr", action="store_true",
                    help="Open a pull request instead of pushing (use for a repo you do not own)")
    args = ap.parse_args()

    pairs = collect(args.models_dir)
    if not any(name.endswith(".pt") for _, name in pairs):
        sys.exit(f"No neural_heuristic_seed*.pt files in {args.models_dir}; run scripts/train_final.py first.")
    if not os.path.exists(os.path.join(args.models_dir, "README.md")):
        print("WARNING: models/README.md missing; run scripts/write_models_readme.py first.")
    for src, dst in pairs:
        print(f"{'would upload' if args.dry_run else 'uploading'}: {src} -> {args.repo_id}/{dst} "
              f"({os.path.getsize(src) / 1e6:.2f} MB)")
    if args.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    if not args.create_pr:
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    for src, dst in pairs:
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=args.repo_id,
                        repo_type="model", commit_message=f"Add {dst}",
                        create_pr=args.create_pr)
    if args.create_pr:
        print(f"Done: pull request(s) opened; the owner must merge them at https://huggingface.co/{args.repo_id}/discussions")
    else:
        print(f"Done: https://huggingface.co/{args.repo_id}/tree/main")


if __name__ == "__main__":
    main()
