"""
K-fold cross-validation driven by an experiment config.

    from gmnbind.cross_validation import cross_validate
    folds, summary = cross_validate("configs/gmn.yaml")

or from the command line:

    python -m gmnbind.cross_validation configs/gmn.yaml [--cv-dir data/index/splits/v2020_casf_cv5]

The model is selected by `model.model_type` in the config ("gmn" or "baseline").
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
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from gmnbind.data.datamodule import ComplexDataModule
from gmnbind.models.baseline import BaselineGNN
from gmnbind.models.gmn import GraphMatchingNetwork

torch.set_float32_matmul_precision("medium")

REPO_ROOT = Path(__file__).resolve().parents[2]      # src/gmnbind/cross_validation.py -> repo root
DEFAULT_CV_DIR = "data/index/splits/v2020_casf_cv5"

# model_type in the config -> model class
MODELS = {"baseline": BaselineGNN, "gmn": GraphMatchingNetwork}


# ---------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------

def _resolve(path):
    """Absolute paths as-is, relative ones from the repo root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(config):
    """Accept a path to a YAML file or an already loaded dict."""
    if isinstance(config, dict):
        return config
    return yaml.safe_load(_resolve(config).read_text())


def model_kwargs(model_cfg):
    """Model constructor arguments from the `model` block of an experiment config."""
    return dict(
        token_dim=model_cfg["token_dim"],
        node_embedding_dims=model_cfg["node_embedding_dims"],
        edge_embedding_dims=model_cfg["edge_embedding_dims"],
        num_layers=model_cfg["num_layers"],
        dropout_p=model_cfg["dropout_p"],
        normalization=model_cfg["normalization"],
        lr=model_cfg["lr"],
        max_lr=model_cfg["max_lr"],
        min_lr=model_cfg["min_lr"],
        l2_norm=model_cfg["l2_norm"],
        warmup_step_percentage=model_cfg["warmup_percentage"],
        use_dynamic_schedule=model_cfg["use_dynamic_schedule"],
    )


def build_model(model_cfg):
    """Fresh, untrained model; model_type selects the class."""
    model_type = model_cfg["model_type"]
    if model_type not in MODELS:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {list(MODELS)}")
    return MODELS[model_type](**model_kwargs(model_cfg))


# ---------------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------------

def _run_fold(config, fold_dir, out_dir):
    """Train one fold and return the validation metrics of the epoch with the lowest val_loss."""
    pl.seed_everything(config["data"]["seed"])          # same init in every fold: only the data change

    data_module = ComplexDataModule(
        processed_dir=_resolve(config["data"]["processed_dir"]),
        split_dir=fold_dir,
        batch_size=config["training"]["batch_size"],
    )
    model = build_model(config["model"])                # fresh model for every fold

    logger = CSVLogger(save_dir=out_dir.parent, name=out_dir.name, version=fold_dir.name)
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        precision="bf16-mixed",
        log_every_n_steps=10,
        logger=logger,
        callbacks=[
            # save_top_k=0: no model saved (metrics are read from metrics.csv);
            # it also prevents Lightning from adding its default checkpoint callback
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=0),
            EarlyStopping(monitor="val_loss", patience=config["training"]["patience"], mode="min"),
        ],
    )
    trainer.fit(model, datamodule=data_module)

    # all the val metrics of the epoch with the lowest val_loss (same row of metrics.csv)
    metrics = pd.read_csv(Path(logger.log_dir) / "metrics.csv")
    val_rows = metrics.dropna(subset=["val_loss"])
    best = val_rows.loc[val_rows["val_loss"].idxmin()]
    val_cols = [c for c in val_rows.columns if c.startswith("val_")]
    return {"fold": fold_dir.name,
            "best_epoch": int(best["epoch"]),
            "n_epochs": int(val_rows["epoch"].max()) + 1,
            **{c: float(best[c]) for c in val_cols}}


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
        rows.append(_run_fold(config, fold_dir, out_dir))
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