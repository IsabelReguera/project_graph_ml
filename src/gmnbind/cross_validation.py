"""
K-fold cross-validation driven by an experiment config.

    from gmnbind.cross_validation import cross_validate
    folds, summary = cross_validate("configs/gmn.yaml")

or from the command line:

    python -m gmnbind.cross_validation configs/gmn.yaml [--cv-dir data/index/splits/v2020_casf_cv5]

The model is selected by `model.model_type` in the config ("gmn" or "baseline").
Each fold is trained with gmnbind.training.train (no checkpoint, no LR logging).
For each fold a fresh model is trained with early stopping on val_loss. The fold's result is the row of
metrics.csv with the lowest val_loss: all the metrics of a fold come from that same epoch.
No checkpoint is saved.

Outputs in <log_dir>/<experiment_name>_cv/:
    fold_i/            metrics.csv, hparams.yaml of each fold
    cv_folds.csv       one row per fold (best epoch + val metrics), updated after every fold
    cv_summary.json    mean and std of every metric over the folds
"""
import warnings
warnings.filterwarnings("ignore", message=".*torch.jit.script.*is deprecated.*", category=FutureWarning)

import argparse
import json

import pandas as pd

# build_model / load_config are re-exported: notebooks import them from here
from gmnbind.training import _resolve, build_model, load_config, train  # noqa: F401

DEFAULT_CV_DIR = "data/index/splits/v2020_casf_cv5"


# ---------------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------------

def cross_validate(config, cv_dir=DEFAULT_CV_DIR, out_dir=None):
    """
    Args:
        config:  experiment config (path to the YAML file or loaded dict).
        cv_dir:  folder with fold_0 ... fold_K-1, each with train.txt and val.txt.
        out_dir: where logs and results are saved (default: <log_dir>/<experiment_name>_cv).

    Returns:
        folds:   DataFrame, one row per fold (best epoch + val metrics)
        summary: dict {metric: {"mean": ..., "std": ...}}  (std = sample std, N-1)
    """
    config = load_config(config)
    model_type = config["model"]["model_type"]
    experiment = config["model"]["experiment_name"]

    cv_dir = _resolve(cv_dir)
    out_dir = _resolve(out_dir) if out_dir else _resolve(config["data"]["log_dir"]) / f"{experiment}_cv"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(p for p in cv_dir.iterdir() if p.is_dir() and p.name.startswith("fold_"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* folders in {cv_dir}")
    print(f"{model_type} [{experiment}]: {len(fold_dirs)}-fold CV on {cv_dir.name}")

    rows = []
    for fold_dir in fold_dirs:
        print(f"\n===== {fold_dir.name} =====")
        # one training per fold: no checkpoint, no LR logging; the fold result is read from metrics.csv
        result = train(config, split_dir=fold_dir, log_dir=out_dir.parent, name=out_dir.name,
                       version=fold_dir.name, save_checkpoint=False, lr_monitor=True)
        rows.append({"fold": fold_dir.name, "best_epoch": result["best_epoch"], "n_epochs": result["n_epochs"],
                     **{k: v for k, v in result.items() if k.startswith("val_")}})
        pd.DataFrame(rows).to_csv(out_dir / "cv_folds.csv", index=False)   # saved after every fold

    folds = pd.DataFrame(rows)
    metric_cols = [c for c in folds.columns if c.startswith("val_")]
    summary = {c: {"mean": float(folds[c].mean()), "std": float(folds[c].std())} for c in metric_cols}

    (out_dir / "cv_summary.json").write_text(json.dumps(
        {"model_type": model_type, "experiment": experiment, "cv_dir": cv_dir.name,
         "n_folds": len(fold_dirs), "config": config, "metrics": summary}, indent=2))

    print(f"\n{model_type} [{experiment}]")
    for c, s in summary.items():
        print(f"  {c:15s} {s['mean']:.4f} ± {s['std']:.4f}")
    return folds, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="experiment config, e.g. configs/gmn.yaml")
    parser.add_argument("--cv-dir", default=DEFAULT_CV_DIR)
    args = parser.parse_args()
    cross_validate(args.config, cv_dir=args.cv_dir)