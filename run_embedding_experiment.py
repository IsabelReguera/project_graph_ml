"""
Ablation: FeatureTokenizer vs DenseFeatureEmbedding.

Grid: {baseline, gmn} x {tokenizer, dense} x 5 seeds = 20 runs.

Output in experiment_embedding_dense/:
  - runs_detail.csv  one row per run (full metrics)
  - runs_summary.csv mean +/- std grouped by (model, embedding)
  - runs/<name>/     lightning logs + best.ckpt per run
"""

import os
import sys
import copy
import yaml
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import CSVLogger


class EpochPrinter(Callback):
    """Stampa una riga per epoca, al posto della progress bar."""
    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        epoch = trainer.current_epoch
        tr = m.get("train_loss")
        vl = m.get("val_loss")
        va = m.get("val_mae")
        vr = m.get("val_pearson_r")
        def fmt(v): return f"{float(v):.4f}" if v is not None else "   -   "
        print(f"  epoch {epoch:3d} | train_loss {fmt(tr)} | val_loss {fmt(vl)} | val_mae {fmt(va)} | val_r {fmt(vr)}", flush=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from baseline import BaselineGNN
from graphmatchingnetwork import GraphMatchingNetwork
from utils import ComplexDataModule, DATASET_SCHEMAS

torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
OUT_DIR          = os.path.join(PROJECT_ROOT, "experiment_embedding_dense")
RUNS_DIR         = os.path.join(OUT_DIR, "runs")
SEEDS            = [42, 47, 51, 53, 59]
MODEL_TYPES      = ["baseline", "gmn"]
EMBEDDING_TYPES  = ["tokenizer", "dense"]
# Optional override of config.data.limit for faster debugging (None = respect config)
LIMIT_OVERRIDE   = None

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Oversmoothing metrics (computed post-training on a validation batch)
# ---------------------------------------------------------------------------
@torch.no_grad()
def mad(x: torch.Tensor, batch: torch.Tensor) -> float:
    """Mean Average Distance (1 - cosine) between node pairs within each graph."""
    vals = []
    for g in batch.unique():
        xg = x[batch == g]
        n = xg.size(0)
        if n < 2:
            continue
        xn = torch.nn.functional.normalize(xg, dim=-1)
        sim = xn @ xn.T
        off = (sim.sum() - sim.diag().sum()) / (n * (n - 1))
        vals.append(1.0 - off.item())
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def dirichlet(x: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Sum of squared differences across edges, averaged."""
    src, dst = edge_index
    return (x[src] - x[dst]).pow(2).sum(dim=-1).mean().item()


@torch.no_grad()
def effective_rank(x: torch.Tensor) -> float:
    """Stable rank  (sum sigma_i)^2 / sum sigma_i^2."""
    s = torch.linalg.svdvals(x.float())
    return (s.sum() ** 2 / (s ** 2).sum()).item()


def feature_group_norms(encoder, schema, embedding_type, cat_emb_dim):
    """
    Norm of first-Linear weights, grouped by (continuous, categorical) input columns.
    Used to check if the MLP ignores continuous features (expected risk with dense).
    """
    W = encoder.mlp[0].weight.detach()
    offset = 0
    cont_cols, cat_cols = [], []
    for i, feat in enumerate(schema):
        if embedding_type == "tokenizer":
            proj = encoder.projectors[i]
            size = proj.out_features if isinstance(proj, torch.nn.Linear) else proj.embedding_dim
        else:  # dense
            size = cat_emb_dim if feat["type"] == "categorical" else 1
        (cat_cols if feat["type"] == "categorical" else cont_cols).extend(
            range(offset, offset + size)
        )
        offset += size
    cont_norm = W[:, cont_cols].norm().item() if cont_cols else 0.0
    cat_norm  = W[:, cat_cols].norm().item()  if cat_cols  else 0.0
    return cont_norm, cat_norm


def extract_last_node_embeddings(model, batch, model_type):
    """
    Register a forward hook on the LAST message-passing layer of each branch,
    run one forward on `batch`, return the post-GNN node embeddings along
    with edge_index and batch assignments for both protein and ligand.
    """
    prot_batch, lig_batch = batch
    captured = {}

    if model_type == "baseline":
        h1 = model.prot_gnn_layers[-1].register_forward_hook(
            lambda m, i, o: captured.setdefault("prot", o.detach())
        )
        h2 = model.lig_gnn_layers[-1].register_forward_hook(
            lambda m, i, o: captured.setdefault("lig", o.detach())
        )
        hooks = [h1, h2]
    else:  # gmn: the layer returns (prot_x, lig_x[, attn...])
        h = model.gmn_layers[-1].register_forward_hook(
            lambda m, i, o: captured.setdefault("gmn_out", o)
        )
        hooks = [h]

    model.eval()
    with torch.no_grad():
        model(prot_batch, lig_batch)
    for h in hooks:
        h.remove()

    if model_type == "baseline":
        x_prot = captured["prot"]
        x_lig  = captured["lig"]
    else:
        out = captured["gmn_out"]
        x_prot = out[0].detach()
        x_lig  = out[1].detach()

    return (
        x_prot, x_lig,
        prot_batch.edge_index, lig_batch.edge_index,
        prot_batch.batch, lig_batch.batch,
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(model_type, embedding_type, seed, base_cfg):
    pl.seed_everything(seed, workers=True)
    cfg = copy.deepcopy(base_cfg)

    experiment_name = f"{model_type}_{embedding_type}_seed{seed}"
    print(f"\n===== {experiment_name} =====")

    logger = CSVLogger(save_dir=RUNS_DIR, name=experiment_name, version=0)

    data_module = ComplexDataModule(
        processed_dir=cfg["data"]["processed_dir"],
        csv_path=cfg["data"]["csv_path"],
        batch_size=cfg["training"]["batch_size"],
        limit=LIMIT_OVERRIDE if LIMIT_OVERRIDE is not None else cfg["data"].get("limit"),
        train_val_split=0.2,
        seed=seed,
    )

    model_cls = BaselineGNN if model_type == "baseline" else GraphMatchingNetwork
    model = model_cls(
        token_dim=cfg["model"]["token_dim"],
        node_embedding_dims=cfg["model"]["node_embedding_dims"],
        edge_embedding_dims=cfg["model"]["edge_embedding_dims"],
        num_layers=cfg["model"]["num_layers"],
        lr=cfg["model"]["lr"],
        max_lr=cfg["model"]["max_lr"],
        min_lr=cfg["model"]["min_lr"],
        warmup_step_percentage=cfg["model"]["warmup_percentage"],
        l2_norm=cfg["model"]["l2_norm"],
        dropout_p=cfg["model"]["dropout_p"],
        normalization=cfg["model"]["normalization"],
        use_dynamic_schedule=cfg["model"]["use_dynamic_schedule"],
        embedding_type=embedding_type,
        cat_emb_dim=cfg["model"]["cat_emb_dim"],
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=logger.log_dir, filename="best",
        monitor="val_loss", mode="min", save_top_k=1,
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="val_loss", patience=cfg["training"]["patience"], mode="min"),
        EpochPrinter(),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        precision="bf16-mixed",
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=data_module)

    # --- metrics.csv analysis ---
    metrics = pd.read_csv(os.path.join(logger.log_dir, "metrics.csv"))
    val   = metrics.dropna(subset=["val_loss"]).sort_values("epoch")
    train = metrics.dropna(subset=["train_loss"]).sort_values("epoch")

    best_row       = val.loc[val["val_loss"].idxmin()]
    best_epoch     = int(best_row["epoch"])
    best_val_loss  = float(best_row["val_loss"])
    best_val_mae   = float(best_row.get("val_mae", float("nan")))
    best_val_r     = float(best_row.get("val_pearson_r", float("nan")))

    tr_at_best = train[train["epoch"] <= best_epoch]
    train_loss_at_best = float(tr_at_best["train_loss"].iloc[-1]) if not tr_at_best.empty else float("nan")
    gen_gap = train_loss_at_best - best_val_loss
    smooth_std = float(val["val_loss"].diff().dropna().std())

    n_params = sum(p.numel() for p in model.parameters())

    # --- post-training analysis on best checkpoint ---
    best_model = model_cls.load_from_checkpoint(ckpt_cb.best_model_path)
    best_model.eval()
    data_module.setup("fit")  # ensure val_ds is ready
    batch = next(iter(data_module.val_dataloader()))
    x_p, x_l, ep, el, bp, bl = extract_last_node_embeddings(best_model, batch, model_type)

    mad_p,  mad_l  = mad(x_p, bp),         mad(x_l, bl)
    dir_p,  dir_l  = dirichlet(x_p, ep),   dirichlet(x_l, el)
    rank_p, rank_l = effective_rank(x_p),  effective_rank(x_l)

    wc_p, wk_p = feature_group_norms(
        best_model.tokenizer.protein_node_enc,
        DATASET_SCHEMAS["protein_node"], embedding_type, cfg["model"]["cat_emb_dim"],
    )
    wc_l, wk_l = feature_group_norms(
        best_model.tokenizer.ligand_node_enc,
        DATASET_SCHEMAS["ligand_node"], embedding_type, cfg["model"]["cat_emb_dim"],
    )

    row = {
        "model": model_type,
        "embedding": embedding_type,
        "seed": seed,
        "params": n_params,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "best_val_r": best_val_r,
        "gen_gap": gen_gap,
        "best_epoch": best_epoch,
        "smooth_std": smooth_std,
        "mad_prot": mad_p, "mad_lig": mad_l,
        "dirichlet_prot": dir_p, "dirichlet_lig": dir_l,
        "eff_rank_prot": rank_p, "eff_rank_lig": rank_l,
        "w_norm_cont_prot": wc_p, "w_norm_cat_prot": wk_p,
        "w_norm_cont_lig": wc_l, "w_norm_cat_lig": wk_l,
    }

    pd.DataFrame([row]).to_csv(os.path.join(logger.log_dir, "run_row.csv"), index=False)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with open(BASE_CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    # fields the base config may lack but train requires
    base_cfg.setdefault("model", {})
    base_cfg["model"].setdefault("cat_emb_dim", 8)
    base_cfg["model"].setdefault("embedding_type", "tokenizer")

    rows = []
    for model_type in MODEL_TYPES:
        for embedding_type in EMBEDDING_TYPES:
            for seed in SEEDS:
                try:
                    row = run_single(model_type, embedding_type, seed, base_cfg)
                    rows.append(row)
                    # save progressively, so partial results survive a crash
                    pd.DataFrame(rows).to_csv(
                        os.path.join(OUT_DIR, "runs_detail.csv"), index=False
                    )
                except Exception as e:
                    print(f"[FAIL] {model_type}/{embedding_type}/seed{seed}: {e}")

    detail = pd.DataFrame(rows)
    if detail.empty:
        print("No successful runs.")
        return

    detail.to_csv(os.path.join(OUT_DIR, "runs_detail.csv"), index=False)

    # --- summary table: mean +/- std grouped by (model, embedding) ---
    numeric_cols = [c for c in detail.columns if c not in ("model", "embedding", "seed")]
    grp = detail.groupby(["model", "embedding"])[numeric_cols]
    mean = grp.mean().add_suffix("_mean")
    std  = grp.std().add_suffix("_std")
    summary = pd.concat([mean, std], axis=1).sort_index(axis=1).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "runs_summary.csv"), index=False)

    print("\n=== DONE ===")
    print(f"detail  -> {os.path.join(OUT_DIR, 'runs_detail.csv')}")
    print(f"summary -> {os.path.join(OUT_DIR, 'runs_summary.csv')}")


if __name__ == "__main__":
    main()
