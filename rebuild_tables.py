"""
Rebuild runs_detail.csv and runs_summary.csv from already-trained checkpoints.

Reads every experiment_embedding_dense/runs/<model>_<embedding>_seed<N>/version_0/
folder, loads best.ckpt + metrics.csv, computes post-training metrics
(MAD, Dirichlet, effective rank, feature-group norms), writes:
  experiment_embedding_dense/runs_detail.csv
  experiment_embedding_dense/runs_summary.csv

Use this if the main experiment script crashed in the post-training analysis
block but the training itself (and best.ckpt) completed.
"""

import os
import re
import sys
import glob
import yaml
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from baseline import BaselineGNN
from graphmatchingnetwork import GraphMatchingNetwork
from utils import ComplexDataModule, DATASET_SCHEMAS

OUT_DIR  = os.path.join(PROJECT_ROOT, "experiment_embedding_dense")
RUNS_DIR = os.path.join(OUT_DIR, "runs")
CONFIG   = os.path.join(PROJECT_ROOT, "config.yaml")

RUN_NAME_RE = re.compile(r"^(baseline|gmn)_(tokenizer|dense)_seed(\d+)$")


# -------------------- Metrics (same as experiment script) --------------------
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


# -------------------- Process one run folder --------------------
def process_run(run_dir, cfg):
    name = os.path.basename(run_dir)
    m = RUN_NAME_RE.match(name)
    if not m:
        return None
    model_type, embedding_type, seed = m.group(1), m.group(2), int(m.group(3))

    version_dirs = sorted(glob.glob(os.path.join(run_dir, "version_*")))
    if not version_dirs:
        print(f"[skip] no version_* in {run_dir}")
        return None
    vdir = version_dirs[-1]

    ckpt_path = os.path.join(vdir, "best.ckpt")
    metrics_path = os.path.join(vdir, "metrics.csv")
    if not os.path.exists(ckpt_path) or not os.path.exists(metrics_path):
        print(f"[skip] missing ckpt/metrics in {vdir}")
        return None

    print(f"[process] {name}")

    # --- metrics.csv ---
    metrics = pd.read_csv(metrics_path)
    val   = metrics.dropna(subset=["val_loss"]).sort_values("epoch")
    train = metrics.dropna(subset=["train_loss"]).sort_values("epoch")
    best_row       = val.loc[val["val_loss"].idxmin()]
    best_epoch     = int(best_row["epoch"])
    best_val_loss  = float(best_row["val_loss"])
    best_val_mae   = float(best_row.get("val_mae", float("nan")))
    best_val_r     = float(best_row.get("val_pearson_r", float("nan")))
    tr_at_best     = train[train["epoch"] <= best_epoch]
    train_at_best  = float(tr_at_best["train_loss"].iloc[-1]) if not tr_at_best.empty else float("nan")
    gen_gap        = train_at_best - best_val_loss
    smooth_std     = float(val["val_loss"].diff().dropna().std())

    # --- load model ---
    model_cls = BaselineGNN if model_type == "baseline" else GraphMatchingNetwork
    model = model_cls.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    # --- data loader (val split; seed from the run name) ---
    data_module = ComplexDataModule(
        processed_dir=cfg["data"]["processed_dir"],
        csv_path=cfg["data"]["csv_path"],
        batch_size=cfg["training"]["batch_size"],
        limit=cfg["data"].get("limit"),
        train_val_split=0.2,
        seed=seed,
    )
    data_module.setup("fit")
    batch = next(iter(data_module.val_dataloader()))

    # --- post-training metrics ---
    x_p, x_l, ep, el, bp, bl = extract_last_node_embeddings(model, batch, model_type)
    mad_p,  mad_l  = mad(x_p, bp),         mad(x_l, bl)
    dir_p,  dir_l  = dirichlet(x_p, ep),   dirichlet(x_l, el)
    rank_p, rank_l = effective_rank(x_p),  effective_rank(x_l)

    cat_emb_dim = cfg["model"].get("cat_emb_dim", 8)
    wc_p, wk_p = feature_group_norms(
        model.tokenizer.protein_node_enc,
        DATASET_SCHEMAS["protein_node"], embedding_type, cat_emb_dim,
    )
    wc_l, wk_l = feature_group_norms(
        model.tokenizer.ligand_node_enc,
        DATASET_SCHEMAS["ligand_node"], embedding_type, cat_emb_dim,
    )

    return {
        "model": model_type, "embedding": embedding_type, "seed": seed,
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
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["model"].setdefault("cat_emb_dim", 8)
    cfg["model"].setdefault("embedding_type", "tokenizer")

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "*"))):
        if not os.path.isdir(run_dir):
            continue
        try:
            row = process_run(run_dir, cfg)
            if row is not None:
                rows.append(row)
        except Exception as e:
            print(f"[FAIL] {os.path.basename(run_dir)}: {e}")

    if not rows:
        print("No runs processed.")
        return

    detail = pd.DataFrame(rows)
    detail.to_csv(os.path.join(OUT_DIR, "runs_detail.csv"), index=False)

    num = [c for c in detail.columns if c not in ("model", "embedding", "seed")]
    grp = detail.groupby(["model", "embedding"])[num]
    summary = pd.concat(
        [grp.mean().add_suffix("_mean"), grp.std().add_suffix("_std")],
        axis=1,
    ).sort_index(axis=1).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "runs_summary.csv"), index=False)

    print(f"\n=== DONE ===")
    print(f"detail  -> {os.path.join(OUT_DIR, 'runs_detail.csv')}   ({len(detail)} rows)")
    print(f"summary -> {os.path.join(OUT_DIR, 'runs_summary.csv')} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
