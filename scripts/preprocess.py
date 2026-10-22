"""
Build protein and ligand graphs for every complex in the index and save them as .pt files.

Versioning is manual: the feature config has a `version` field and the graphs are saved
in data/processed/{version}/. On the first run a meta.json is written there with a snapshot
of the config, the git commit and the creation date. On later runs:
  - same config      -> resume (complexes already processed are skipped)
  - different config -> stop with an error (bump `version` in the config)

At the end meta.json is updated with the counts and the list of excluded complexes
(status "excluded") and of ligands built from SMILES instead of SDF (status "fallback").
meta.json is versioned in git; the .pt files are not.

NOTE: the check compares the config only. If you change the featurization code
(protein.py, ligand.py, pdbbind_dataset.py), bump `version` yourself.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --config configs/data/complex_features.yaml --limit 20
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from gmnbind.data.pdbbind_dataset import PDBbindDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def now():
    return datetime.now().isoformat(timespec="seconds")


def git_commit():
    """Current commit hash, with '-dirty' if there are uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def check_or_create_meta(output_dir, feat_cfg):
    """First run: create meta.json. Later runs: stop if the config changed."""
    meta_path = output_dir / "meta.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["config"] != feat_cfg:
            raise RuntimeError(
                f"{output_dir} was generated with a different feature config.\n"
                f"Bump `version` in the config (e.g. v{int(feat_cfg['version'][1:]) + 1}) "
                f"or delete the folder."
            )
        print(f"meta.json matches, resuming in {output_dir}")
        return meta

    if output_dir.exists() and any(output_dir.glob("*.pt")):
        print(f"WARNING: {output_dir} already contains .pt files but no meta.json. "
              f"They are assumed to come from this config.")

    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": feat_cfg["version"],
        "created": now(),
        "git_commit": git_commit(),
        "config": feat_cfg,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def collect_excluded(output_dir, index_df, failures, previous):
    """
    Excluded = complexes of the index without both .pt files (checked on disk), with the reason
    from this run or from a previous one. Fallbacks = ligands built from SMILES, kept across runs.
    """
    prev_reasons = {(e["pdb_code"], e["mode"]): e for e in previous}
    new_reasons = {(f["pdb_code"], f["mode"]): f for f in failures}
    reasons = {**prev_reasons, **new_reasons}

    entries = []
    for code in index_df["pdb_code"]:
        for mode in ("protein", "ligand"):
            exists = (output_dir / f"{mode}_{code}.pt").exists()
            r = reasons.get((code, mode))
            if not exists:
                entries.append({"pdb_code": code, "mode": mode, "status": "excluded",
                                "reason": r["reason"] if r and r["status"] == "excluded"
                                else "missing from a previous run"})
            elif r and r["status"] == "fallback":
                entries.append(r)

    excluded_codes = {e["pdb_code"] for e in entries if e["status"] == "excluded"}
    n_ok = len(index_df) - len(excluded_codes)
    return entries, n_ok, len(excluded_codes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data/complex_features.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N complexes of the index")
    args = parser.parse_args()

    feat_cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    index_path = REPO_ROOT / feat_cfg["index"]
    structures_dir = REPO_ROOT / feat_cfg["structures_dir"]
    output_dir = PROCESSED_DIR / feat_cfg["version"]
    print(f"Feature version: {feat_cfg['version']} -> {output_dir}")

    meta = check_or_create_meta(output_dir, feat_cfg)

    index_df = pd.read_csv(index_path)
    if args.limit is not None:
        index_df = index_df.iloc[:args.limit]

    common = dict(
        root=DATA_DIR,
        csv_path=index_path,
        output_dir=output_dir,
        structures_dir=structures_dir,
        from_local_files=True,
        fallback_to_smiles=feat_cfg["ligand"]["fallback_to_smiles"],
        limit=args.limit,
    )

    print("--- PROTEIN GENERATION ---")
    protein_dataset = PDBbindDataset(mode="protein", **common)
    print("\n--- LIGAND GENERATION ---")
    ligand_dataset = PDBbindDataset(mode="ligand", **common)

    entries, n_ok, n_excl = collect_excluded(
        output_dir, index_df,
        protein_dataset.failures + ligand_dataset.failures,
        meta.get("excluded", []),
    )
    n_fallback = sum(e["status"] == "fallback" for e in entries)

    meta.update({
        "finished": now(),
        "last_run_git_commit": git_commit(),
        "last_run_limit": args.limit,
        "n_complexes": n_ok,
        "n_excluded": n_excl,
        "n_smiles_fallback": n_fallback,
        "excluded": entries,
    })
    (output_dir / ".meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n{n_ok} ok | {n_excl} excluded | {n_fallback} ligands from SMILES")
    print(f"Done. meta.json written to {output_dir}")


if __name__ == "__main__":
    main()
