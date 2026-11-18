"""
Training of one model from an experiment config.

    from gmnbind.training import train
    result = train("configs/gmn.yaml")

Used by scripts/train.py (single run) and by gmnbind.cross_validation (one run per fold).
The model is selected by `model.model_type` in the config ("gmn" or "baseline").
Training uses early stopping on val_loss; the result of a run is the row of metrics.csv
with the lowest val_loss (all the metrics of that same epoch).
"""
import warnings
warnings.filterwarnings("ignore", message=".*torch.jit.script.*is deprecated.*", category=FutureWarning)

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from gmnbind.data.datamodule import ComplexDataModule
from gmnbind.models.baseline import BaselineGNN
from gmnbind.models.gmn import GraphMatchingNetwork

torch.set_float32_matmul_precision("medium")      # to improve performance on some GPUs

REPO_ROOT = Path(__file__).resolve().parents[2]      # src/gmnbind/training.py -> repo root

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


def best_epoch_metrics(metrics_csv):
    """Row of metrics.csv with the lowest val_loss: best epoch, number of epochs and all val_* metrics."""
    metrics = pd.read_csv(metrics_csv)
    val_rows = metrics.dropna(subset=["val_loss"])
    best = val_rows.loc[val_rows["val_loss"].idxmin()]
    val_cols = [c for c in val_rows.columns if c.startswith("val_")]
    return {"best_epoch": int(best["epoch"]),
            "n_epochs": int(val_rows["epoch"].max()) + 1,
            **{c: float(best[c]) for c in val_cols}}


# ---------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------

def train(config, split_dir=None, log_dir=None, name=None, version=None,
          save_checkpoint=True, lr_monitor=True):
    """
    Train one model.

    Args:
        config:          experiment config (path to the YAML file or loaded dict).
        split_dir:       split folder (default: data.split of the config).
        log_dir, name:   logs go to <log_dir>/<name>/<version>/ (default: data.log_dir, model.experiment_name).
        version:         log subfolder (default: version_0, version_1, ... assigned by Lightning).
        save_checkpoint: save the model of the best epoch as best.ckpt.
        lr_monitor:      log the learning rate at every step.

    Returns:
        dict with log_dir, version, best_model_path (None if not saved), best_epoch, n_epochs
        and all the val_* metrics of the best epoch.
    """
    config = load_config(config)
    pl.seed_everything(config["data"]["seed"])                  # for reproducibility

    logger = CSVLogger(save_dir=_resolve(log_dir or config["data"]["log_dir"]),
                       name=name or config["model"]["experiment_name"], version=version)

    checkpoint = ModelCheckpoint(dirpath=logger.log_dir, filename="best", monitor="val_loss", mode="min",
                                 # save_top_k=0: no model saved; it also prevents Lightning
                                 # from adding its default checkpoint callback
                                 save_top_k=1 if save_checkpoint else 0)
    callbacks = [checkpoint,
                 EarlyStopping(monitor="val_loss", patience=config["training"]["patience"], mode="min")]
    if lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    data_module = ComplexDataModule(
        processed_dir=_resolve(config["data"]["processed_dir"]),
        split_dir=_resolve(split_dir or config["data"]["split"]),
        batch_size=config["training"]["batch_size"],
    )
    model = build_model(config["model"])                        # fresh model

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        precision="bf16-mixed",
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=data_module)

    return {"log_dir": logger.log_dir,
            "version": logger.version,
            "best_model_path": checkpoint.best_model_path if save_checkpoint else None,
            **best_epoch_metrics(Path(logger.log_dir) / "metrics.csv")}
