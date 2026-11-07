"""
Orchestrates the PLAGCA hyperparameters ablation.

Grid: 3 configurations (A, B, C) x 2 models (baseline, gmn) x 5 seeds = 30 runs.
Output in experiment_parameters_PLAGCA/:
  - runs_detail.csv   one row per run
  - runs_summary.csv  mean +/- std grouped by (config, model)
  - runs/<config>_<model>_seed<N>/  Lightning logs + best.ckpt per run

Resume supported: already-completed (config, model, seed) triples are skipped
if runs_detail.csv already contains them.
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

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, HERE)

from baseline import BaselineGNN
from graphmatchingnetwork import GraphMatchingNetwork
from baseline_paper import BaselinePaperGNN
from graphmatchingnetwork_paper import GMNPaperStyle

from utils import ComplexDataModule, DATASET_SCHEMAS
from filtered_data_module import FilteredComplexDataModule

torch.set_float32_matmul_precision("medium")

OUT_DIR  = HERE
RUNS_DIR = os.path.join(OUT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

CONFIGS = [
    ("A", os.path.join(HERE, "config_A.yaml")),
    ("B", os.path.join(HERE, "config_B.yaml")),
    ("C", os.path.join(HERE, "config_C.yaml")),
]
MODELS = ["baseline", "gmn"]
SEEDS  = [42, 47, 51, 53, 59]


# -------------------- Metrics --------------------
@torch.no_grad()
def mad(x, batch):
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
def dirichlet(x, edge_index):
    if edge_index.numel() == 0:
        return float("nan")
    src, dst = edge_index
    return (x[src] - x[dst]).pow(2).sum(dim=-1).mean().item()


@torch.no_grad()
def effective_rank(x):
    s = torch.linalg.svdvals(x.float())
    return (s.sum() ** 2 / (s ** 2).sum()).item()


def feature_group_norms(encoder, schema, embedding_type, cat_emb_dim):
    W = encoder.mlp[0].weight.detach()
    offset = 0
    cont_cols, cat_cols = [], []
    for i, feat in enumerate(schema):
        if embedding_type == "tokenizer":
            proj = encoder.projectors[i]
            size = proj.out_features if isinstance(proj, torch.nn.Linear) else proj.embedding_dim
        else:
            size = cat_emb_dim if feat["type"] == "categorical" else 1
        (cat_cols if feat["type"] == "categorical" else cont_cols).extend(
            range(offset, offset + size)
        )
        offset += size
    cont_norm = W[:, cont_cols].norm().item() if cont_cols else 0.0
    cat_norm  = W[:, cat_cols].norm().item()  if cat_cols  else 0.0
    return cont_norm, cat_norm


def extract_last_node_embeddings(model, batch, model_type):
    device = next(model.parameters()).device
    prot_batch, lig_batch = batch
    prot_batch = prot_batch.to(device)
    lig_batch  = lig_batch.to(device)
    captured = {}

    if model_type == "baseline":
        h1 = model.prot_gnn_layers[-1].register_forward_hook(
            lambda m, i, o: captured.setdefault("prot", o.detach())
        )
        h2 = model.lig_gnn_layers[-1].register_forward_hook(
            lambda m, i, o: captured.setdefault("lig", o.detach())
        )
        hooks = [h1, h2]
    else:
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
        x_prot, x_lig = captured["prot"], captured["lig"]
    else:
        out = captured["gmn_out"]
        x_prot, x_lig = out[0].detach(), out[1].detach()

    return (
        x_prot, x_lig,
        prot_batch.edge_index, lig_batch.edge_index,
        prot_batch.batch, lig_batch.batch,
    )


class EpochPrinter(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        epoch = trainer.current_epoch
        def fmt(v): return f"{float(v):.4f}" if v is not None else "   -   "
        print(
            f"  epoch {epoch:3d} | train_loss {fmt(m.get('train_loss'))} "
            f"| val_loss {fmt(m.get('val_loss'))} "
            f"| val_mae {fmt(m.get('val_mae'))} "
            f"| val_r {fmt(m.get('val_pearson_r'))}",
            flush=True,
        )


# -------------------- Factories --------------------
def build_model(cfg, model_type):
    variant = cfg["model"].get("model_variant", "default")
    mcfg = cfg["model"]
    kwargs = dict(
        token_dim=mcfg["token_dim"],
        node_embedding_dims=mcfg["node_embedding_dims"],
        edge_embedding_dims=mcfg["edge_embedding_dims"],
        num_layers=mcfg["num_layers"],
        lr=mcfg["lr"],
        max_lr=mcfg["max_lr"],
        min_lr=mcfg["min_lr"],
        warmup_step_percentage=mcfg["warmup_percentage"],
        l2_norm=mcfg["l2_norm"],
        dropout_p=mcfg["dropout_p"],
        normalization=mcfg["normalization"],
        use_dynamic_schedule=mcfg["use_dynamic_schedule"],
        embedding_type=mcfg["embedding_type"],
        cat_emb_dim=mcfg["cat_emb_dim"],
    )
    if variant == "paper":
        cls = BaselinePaperGNN if model_type == "baseline" else GMNPaperStyle
        kwargs["mlp_hidden_sizes"] = mcfg.get("mlp_hidden_sizes")
        return cls(**kwargs)
    else:
        cls = BaselineGNN if model_type == "baseline" else GraphMatchingNetwork
        return cls(**kwargs)


def build_data_module(cfg, seed):
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    gfg = cfg.get("graph_filters", {})
    common = dict(
        processed_dir=dcfg["processed_dir"],
        csv_path=dcfg["csv_path"],
        batch_size=tcfg["batch_size"],
        limit=dcfg.get("limit"),
        train_val_split=0.2,
        seed=seed,
    )
    use_filter = gfg.get("protein_max_distance") is not None or gfg.get("ligand_bonds_only", False)
    if use_filter:
        return FilteredComplexDataModule(
            **common,
            protein_max_distance=gfg.get("protein_max_distance"),
            ligand_bonds_only=gfg.get("ligand_bonds_only", False),
        )
    return ComplexDataModule(**common)


# -------------------- Single run --------------------
def run_single(config_name, config_path, model_type, seed):
    pl.seed_everything(seed, workers=True)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    experiment_name = f"{config_name}_{model_type}_seed{seed}"
    print(f"\n===== {experiment_name} =====", flush=True)

    logger = CSVLogger(save_dir=RUNS_DIR, name=experiment_name, version=0)
    data_module = build_data_module(cfg, seed)
    model = build_model(cfg, model_type)

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
    best_row = val.loc[val["val_loss"].idxmin()]
    best_epoch    = int(best_row["epoch"])
    best_val_loss = float(best_row["val_loss"])
    best_val_mae  = float(best_row.get("val_mae", float("nan")))
    best_val_r    = float(best_row.get("val_pearson_r", float("nan")))
    tr_at_best = train[train["epoch"] <= best_epoch]
    train_at_best = float(tr_at_best["train_loss"].iloc[-1]) if not tr_at_best.empty else float("nan")
    gen_gap = train_at_best - best_val_loss
    smooth_std = float(val["val_loss"].diff().dropna().std())
    n_params = sum(p.numel() for p in model.parameters())

    # --- post-training on best checkpoint ---
    variant = cfg["model"].get("model_variant", "default")
    if variant == "paper":
        cls = BaselinePaperGNN if model_type == "baseline" else GMNPaperStyle
    else:
        cls = BaselineGNN if model_type == "baseline" else GraphMatchingNetwork
    best_model = cls.load_from_checkpoint(ckpt_cb.best_model_path, map_location="cpu")
    best_model.eval()

    data_module.setup("fit")
    batch = next(iter(data_module.val_dataloader()))
    x_p, x_l, ep, el, bp, bl = extract_last_node_embeddings(best_model, batch, model_type)
    mad_p,  mad_l  = mad(x_p, bp),        mad(x_l, bl)
    dir_p,  dir_l  = dirichlet(x_p, ep),  dirichlet(x_l, el)
    rank_p, rank_l = effective_rank(x_p), effective_rank(x_l)

    cat_emb_dim = cfg["model"]["cat_emb_dim"]
    emb_type    = cfg["model"]["embedding_type"]
    wc_p, wk_p = feature_group_norms(
        best_model.tokenizer.protein_node_enc,
        DATASET_SCHEMAS["protein_node"], emb_type, cat_emb_dim,
    )
    wc_l, wk_l = feature_group_norms(
        best_model.tokenizer.ligand_node_enc,
        DATASET_SCHEMAS["ligand_node"], emb_type, cat_emb_dim,
    )

    return {
        "config": config_name, "model": model_type, "seed": seed,
        "params": n_params,
        "best_val_loss": best_val_loss, "best_val_mae": best_val_mae, "best_val_r": best_val_r,
        "gen_gap": gen_gap, "best_epoch": best_epoch, "smooth_std": smooth_std,
        "mad_prot": mad_p, "mad_lig": mad_l,
        "dirichlet_prot": dir_p, "dirichlet_lig": dir_l,
        "eff_rank_prot": rank_p, "eff_rank_lig": rank_l,
        "w_norm_cont_prot": wc_p, "w_norm_cat_prot": wk_p,
        "w_norm_cont_lig": wc_l, "w_norm_cat_lig": wk_l,
    }


# -------------------- Main --------------------
def main():
    detail_path = os.path.join(OUT_DIR, "runs_detail.csv")
    if os.path.exists(detail_path):
        existing = pd.read_csv(detail_path)
        done = {(r.config, r.model, int(r.seed)) for r in existing.itertuples()}
        rows = existing.to_dict("records")
        print(f"[resume] {len(done)} runs already in {detail_path}, skipping them")
    else:
        done = set()
        rows = []

    for config_name, config_path in CONFIGS:
        for model_type in MODELS:
            for seed in SEEDS:
                if (config_name, model_type, seed) in done:
                    print(f"[skip] {config_name}/{model_type}/seed{seed} already done")
                    continue
                try:
                    row = run_single(config_name, config_path, model_type, seed)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(detail_path, index=False)
                except Exception as e:
                    print(f"[FAIL] {config_name}/{model_type}/seed{seed}: {e}")

    detail = pd.DataFrame(rows)
    if detail.empty:
        print("No successful runs."); return
    detail.to_csv(detail_path, index=False)

    num = [c for c in detail.columns if c not in ("config", "model", "seed")]
    grp = detail.groupby(["config", "model"])[num]
    summary = pd.concat(
        [grp.mean().add_suffix("_mean"), grp.std().add_suffix("_std")],
        axis=1,
    ).sort_index(axis=1).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "runs_summary.csv"), index=False)

    print("\n=== DONE ===")
    print(f"detail  -> {detail_path}")
    print(f"summary -> {os.path.join(OUT_DIR, 'runs_summary.csv')}")


if __name__ == "__main__":
    main()
