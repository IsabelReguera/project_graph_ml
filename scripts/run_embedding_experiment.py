"""
Ablation FeatureTokenizer vs DenseFeatureEmbedding, repeated on the v2 dataset.

Grid: {baseline, gmn} x {tokenizer, dense} x SEEDS  (default 5 seeds = 20 runs).

This is the refactored-pipeline version of the original `run_embedding_experiment.py`
(layer_dense_experiment branch). It trains through gmnbind.training.train, so every run
uses the DETERMINISTIC CASF split (data/index/splits/v2020_casf) and the v2 processed
graphs (data/processed/v2). The oversmoothing metrics (MAD, Dirichlet energy, effective
rank) and the first-Linear feature-group norms use the SAME definitions as the v1
experiment, so v1 and v2 are directly comparable.

Note on seeds: with the deterministic split the train/val/test sets are FIXED; a seed
only changes weight initialization and training stochasticity (not the split, unlike the
old v1 experiment where the seed also reshuffled train/val).

Output (self-contained) in experiment_embedding_dense/runs/v2/:
  - runs_detail.csv    one row per run
  - runs_summary.csv   mean +/- std grouped by (model, embedding)
  - <model>_<emb>_seed<N>/version_0/   Lightning logs + best.ckpt + run_row.csv

Resume: completed (model, embedding, seed) triples are read from runs_detail.csv and
skipped, so re-launching recovers from an interruption without wasting GPU time.

Usage (on the VM, from the repo root, gmnbind importable or src/ on the path):
    python scripts/run_embedding_experiment.py
To smoke-test first, temporarily set EPOCHS to a small value (e.g. 3).
"""
import copy
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))   # make gmnbind importable without `pip install -e .`

from gmnbind.training import load_config, train
from gmnbind.data.datamodule import ComplexDataModule
from gmnbind.data.schemas import DATASET_SCHEMAS
from gmnbind.models.baseline import BaselineGNN
from gmnbind.models.gmn import GraphMatchingNetwork

torch.set_float32_matmul_precision("medium")

# --- experiment grid -------------------------------------------------------
DATASET_TAG     = "v2"
BASE_CONFIG     = REPO_ROOT / "configs" / "gmn.yaml"   # hyperparameters shared by all runs
SEEDS           = [42]   # single seed for v2 (full dataset is heavy); swap for 47/51/53/59 to match another v1 seed
MODEL_TYPES     = ["baseline", "gmn"]
EMBEDDING_TYPES = ["tokenizer", "dense"]
EPOCHS          = 1000    # same as the v1 experiment; early stopping does the real cutoff
PATIENCE        = 30

OUT_DIR = REPO_ROOT / "experiment_embedding_dense" / "runs" / DATASET_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
DETAIL_PATH  = OUT_DIR / "runs_detail.csv"
SUMMARY_PATH = OUT_DIR / "runs_summary.csv"

MODEL_CLS = {"baseline": BaselineGNN, "gmn": GraphMatchingNetwork}


# --- oversmoothing metrics (identical definitions to the v1 experiment) ----
@torch.no_grad()
def mad(x, batch):
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
def dirichlet(x, edge_index):
    """Mean squared difference across edges."""
    src, dst = edge_index
    return (x[src] - x[dst]).pow(2).sum(dim=-1).mean().item()


@torch.no_grad()
def effective_rank(x):
    """Stable rank  (sum sigma_i)^2 / sum sigma_i^2."""
    s = torch.linalg.svdvals(x.float())
    return (s.sum() ** 2 / (s ** 2).sum()).item()


def feature_group_norms(encoder, schema, embedding_type, cat_emb_dim):
    """Norm of first-Linear weights, grouped by (continuous, categorical) input columns."""
    W = encoder.mlp[0].weight.detach()
    offset = 0
    cont_cols, cat_cols = [], []
    for i, feat in enumerate(schema):
        if embedding_type == "tokenizer":
            proj = encoder.projectors[i]
            size = proj.out_features if isinstance(proj, torch.nn.Linear) else proj.embedding_dim
        else:  # dense
            size = cat_emb_dim if feat["type"] == "categorical" else 1
        (cat_cols if feat["type"] == "categorical" else cont_cols).extend(range(offset, offset + size))
        offset += size
    cont_norm = W[:, cont_cols].norm().item() if cont_cols else 0.0
    cat_norm = W[:, cat_cols].norm().item() if cat_cols else 0.0
    return cont_norm, cat_norm


@torch.no_grad()
def extract_last_node_embeddings(model, batch, model_type):
    """Forward hook on the LAST message-passing layer of each branch -> post-GNN node embeddings."""
    device = next(model.parameters()).device
    prot_batch, lig_batch = batch
    prot_batch = prot_batch.to(device)
    lig_batch = lig_batch.to(device)
    captured = {}

    if model_type == "baseline":
        h1 = model.prot_gnn_layers[-1].register_forward_hook(
            lambda m, i, o: captured.setdefault("prot", o.detach()))
        h2 = model.lig_gnn_layers[-1].register_forward_hook(
            lambda m, i, o: captured.setdefault("lig", o.detach()))
        hooks = [h1, h2]
    else:  # gmn: the layer returns (prot_x, lig_x[, attn...])
        h = model.gmn_layers[-1].register_forward_hook(
            lambda m, i, o: captured.setdefault("gmn_out", o))
        hooks = [h]

    model.eval()
    model(prot_batch, lig_batch)
    for h in hooks:
        h.remove()

    if model_type == "baseline":
        x_prot, x_lig = captured["prot"], captured["lig"]
    else:
        out = captured["gmn_out"]
        x_prot, x_lig = out[0].detach(), out[1].detach()

    return x_prot, x_lig, prot_batch.edge_index, lig_batch.edge_index, prot_batch.batch, lig_batch.batch


# --- single run ------------------------------------------------------------
def run_single(model_type, embedding_type, seed, base_cfg):
    name = f"{model_type}_{embedding_type}_seed{seed}"
    print(f"\n===== [{DATASET_TAG}] {name} =====", flush=True)

    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["model_type"]      = model_type
    cfg["model"]["embedding_type"]  = embedding_type
    cfg["model"]["experiment_name"] = name
    cfg["data"]["seed"]             = seed
    cfg["training"]["epochs"]       = EPOCHS
    cfg["training"]["patience"]     = PATIENCE

    # Train on the deterministic split through the refactored pipeline.
    # train() resolves relative paths in cfg against the repo root.
    result = train(cfg, log_dir=str(OUT_DIR), name=name, version=0, save_checkpoint=True)
    log_dir = Path(result["log_dir"])

    # --- task metrics from metrics.csv (same extraction logic as v1) ---
    metrics = pd.read_csv(log_dir / "metrics.csv")
    val = metrics.dropna(subset=["val_loss"]).sort_values("epoch")
    tr = metrics.dropna(subset=["train_loss"]).sort_values("epoch")
    best_row      = val.loc[val["val_loss"].idxmin()]
    best_epoch    = int(best_row["epoch"])
    best_val_loss = float(best_row["val_loss"])
    best_val_mae  = float(best_row.get("val_mae", float("nan")))
    best_val_r    = float(best_row.get("val_pearson", float("nan")))   # refactor renamed val_pearson_r -> val_pearson
    tr_at_best = tr[tr["epoch"] <= best_epoch]
    train_loss_at_best = float(tr_at_best["train_loss"].iloc[-1]) if not tr_at_best.empty else float("nan")
    gen_gap = train_loss_at_best - best_val_loss
    smooth_std = float(val["val_loss"].diff().dropna().std())

    # --- post-training oversmoothing analysis on the best checkpoint ---
    model_cls = MODEL_CLS[model_type]
    best_model = model_cls.load_from_checkpoint(result["best_model_path"])
    best_model.to("cuda" if torch.cuda.is_available() else "cpu")
    best_model.eval()
    n_params = sum(p.numel() for p in best_model.parameters())

    dm = ComplexDataModule(
        processed_dir=str(REPO_ROOT / cfg["data"]["processed_dir"]),
        split_dir=str(REPO_ROOT / cfg["data"]["split"]),
        batch_size=cfg["training"]["batch_size"],
    )
    dm.setup("fit")
    batch = next(iter(dm.val_dataloader()))
    x_p, x_l, ep, el, bp, bl = extract_last_node_embeddings(best_model, batch, model_type)

    mad_p, mad_l   = mad(x_p, bp), mad(x_l, bl)
    dir_p, dir_l   = dirichlet(x_p, ep), dirichlet(x_l, el)
    rank_p, rank_l = effective_rank(x_p), effective_rank(x_l)
    wc_p, wk_p = feature_group_norms(best_model.tokenizer.protein_node_enc,
                                     DATASET_SCHEMAS["protein_node"], embedding_type, cfg["model"]["cat_emb_dim"])
    wc_l, wk_l = feature_group_norms(best_model.tokenizer.ligand_node_enc,
                                     DATASET_SCHEMAS["ligand_node"], embedding_type, cfg["model"]["cat_emb_dim"])

    row = {
        "dataset": DATASET_TAG,
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
    pd.DataFrame([row]).to_csv(log_dir / "run_row.csv", index=False)
    return row


# --- main ------------------------------------------------------------------
def main():
    base_cfg = load_config(BASE_CONFIG)
    base_cfg.setdefault("model", {})
    base_cfg["model"].setdefault("cat_emb_dim", 8)

    # Resume: load existing rows and skip already-completed runs.
    if DETAIL_PATH.exists():
        existing = pd.read_csv(DETAIL_PATH)
        done = {(r.model, r.embedding, int(r.seed)) for r in existing.itertuples()}
        rows = existing.to_dict("records")
        print(f"[resume] {len(done)} runs already in {DETAIL_PATH}, skipping them", flush=True)
    else:
        done, rows = set(), []

    for model_type in MODEL_TYPES:
        for embedding_type in EMBEDDING_TYPES:
            for seed in SEEDS:
                if (model_type, embedding_type, seed) in done:
                    print(f"[skip] {model_type}/{embedding_type}/seed{seed} already done", flush=True)
                    continue
                try:
                    rows.append(run_single(model_type, embedding_type, seed, base_cfg))
                    pd.DataFrame(rows).to_csv(DETAIL_PATH, index=False)   # write after every run
                except Exception as e:
                    print(f"[FAIL] {model_type}/{embedding_type}/seed{seed}: {e}", flush=True)
                    traceback.print_exc()

    detail = pd.DataFrame(rows)
    if detail.empty:
        print("No successful runs.")
        return
    detail.to_csv(DETAIL_PATH, index=False)

    # summary: mean +/- std grouped by (model, embedding)
    numeric = [c for c in detail.columns if c not in ("dataset", "model", "embedding", "seed")]
    grp = detail.groupby(["model", "embedding"])[numeric]
    summary = pd.concat([grp.mean().add_suffix("_mean"), grp.std().add_suffix("_std")], axis=1)
    summary.sort_index(axis=1).reset_index().to_csv(SUMMARY_PATH, index=False)

    print(f"\n=== DONE ({DATASET_TAG}) ===", flush=True)
    print(f"detail  -> {DETAIL_PATH}", flush=True)
    print(f"summary -> {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
