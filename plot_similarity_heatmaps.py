"""
Cosine-similarity heatmaps of post-GNN node embeddings.

For each of 4 model/embedding configurations
({baseline, gmn} x {tokenizer, dense}), loads all 5 seed checkpoints,
runs inference on 3 pre-selected protein-ligand complexes, extracts
the node embeddings AFTER the last message-passing layer, and plots
the mean cosine-similarity matrix S = X_norm @ X_norm.T averaged
across the 5 seeds.

Interpretation: more uniform/red heatmap = nodes collapsed together
(stronger oversmoothing). More contrasted heatmap = nodes more
differentiated.

Outputs (two PNG files):
  experiment_embedding_dense/figures/similarity_proteins.png
  experiment_embedding_dense/figures/similarity_ligands.png
"""

import os
import sys
import glob
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from baseline import BaselineGNN
from graphmatchingnetwork import GraphMatchingNetwork

OUT_DIR = os.path.join(PROJECT_ROOT, "experiment_embedding_dense")
RUNS_DIR = os.path.join(OUT_DIR, "runs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 47, 51, 53, 59]
CONFIGS = [
    ("baseline", "tokenizer"),
    ("baseline", "dense"),
    ("gmn",      "tokenizer"),
    ("gmn",      "dense"),
]

# PDB codes: if None, picked automatically from available processed files.
# Set to a list of 3 pdb codes to force a specific choice, e.g.
# PDB_CODES = ["1a1e", "1b40", "2zff"]
PDB_CODES = None


# -------------------- Helpers --------------------
def pick_pdb_codes(n=3):
    """Pick n PDB codes that have BOTH protein_<pdb>.pt and ligand_<pdb>.pt."""
    prot = {os.path.basename(p)[8:-3] for p in glob.glob(os.path.join(PROCESSED, "protein_*.pt"))}
    lig  = {os.path.basename(p)[7:-3]  for p in glob.glob(os.path.join(PROCESSED, "ligand_*.pt"))}
    common = sorted(prot & lig)
    if len(common) < n:
        raise RuntimeError(f"Only {len(common)} complexes available; need {n}.")
    # take 3 complexes spread across the sorted list (so they are not neighbours)
    step = max(1, len(common) // n)
    return common[::step][:n]


def load_complex(pdb):
    """Load a single complex as a 1-graph batch (both protein and ligand)."""
    prot = torch.load(os.path.join(PROCESSED, f"protein_{pdb}.pt"), weights_only=False)
    lig  = torch.load(os.path.join(PROCESSED, f"ligand_{pdb}.pt"),  weights_only=False)
    return Batch.from_data_list([prot]), Batch.from_data_list([lig])


def extract_last_embeddings(model, prot_batch, lig_batch, model_type):
    """Register a forward hook on the last message-passing layer(s),
    run a forward, return post-GNN node embeddings for protein and ligand."""
    device = next(model.parameters()).device
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
        return captured["prot"].cpu(), captured["lig"].cpu()
    out = captured["gmn_out"]
    return out[0].detach().cpu(), out[1].detach().cpu()


def cosine_sim_matrix(X):
    Xn = torch.nn.functional.normalize(X, dim=-1)
    return (Xn @ Xn.T).numpy()


def load_ckpt(model_type, embedding_type, seed):
    """Load the best checkpoint for a (model, embedding, seed) run."""
    run_dir = os.path.join(RUNS_DIR, f"{model_type}_{embedding_type}_seed{seed}")
    ckpt_path = os.path.join(run_dir, "version_0", "best.ckpt")
    if not os.path.exists(ckpt_path):
        return None
    cls = BaselineGNN if model_type == "baseline" else GraphMatchingNetwork
    return cls.load_from_checkpoint(ckpt_path, map_location="cpu")


# -------------------- Main --------------------
def main():
    pdb_codes = PDB_CODES or pick_pdb_codes(3)
    print(f"Using PDB codes: {pdb_codes}")

    # sim_mats[config_idx][pdb_idx] -> (S_prot_mean, S_lig_mean)
    sim_mats = [[None] * len(pdb_codes) for _ in CONFIGS]

    for ci, (model_type, embedding_type) in enumerate(CONFIGS):
        print(f"\n=== {model_type} / {embedding_type} ===")
        # one model per seed, reused across all 3 complexes
        models = []
        for s in SEEDS:
            m = load_ckpt(model_type, embedding_type, s)
            if m is None:
                print(f"  [skip] no ckpt for seed {s}")
                continue
            models.append(m)
        if not models:
            print(f"  [FAIL] no checkpoints for {model_type}/{embedding_type}")
            continue

        for pi, pdb in enumerate(pdb_codes):
            prot_batch, lig_batch = load_complex(pdb)
            S_prot_list, S_lig_list = [], []
            for m in models:
                x_p, x_l = extract_last_embeddings(m, prot_batch, lig_batch, model_type)
                S_prot_list.append(cosine_sim_matrix(x_p))
                S_lig_list.append(cosine_sim_matrix(x_l))
            # all seed matrices share the same node order (same graph file),
            # so a plain element-wise mean is valid.
            sim_mats[ci][pi] = (
                np.mean(np.stack(S_prot_list), axis=0),
                np.mean(np.stack(S_lig_list),  axis=0),
            )
            print(f"  {pdb}: prot {sim_mats[ci][pi][0].shape}, lig {sim_mats[ci][pi][1].shape}")

    # -------------------- Plot --------------------
    col_titles = [f"{mt}\n{et}" for mt, et in CONFIGS]
    for kind, idx in [("proteins", 0), ("ligands", 1)]:
        fig, axes = plt.subplots(
            len(pdb_codes), len(CONFIGS),
            figsize=(3.2 * len(CONFIGS), 3.2 * len(pdb_codes)),
            squeeze=False,
        )
        for pi, pdb in enumerate(pdb_codes):
            for ci in range(len(CONFIGS)):
                ax = axes[pi][ci]
                entry = sim_mats[ci][pi]
                if entry is None:
                    ax.set_axis_off()
                    continue
                S = entry[idx]
                im = ax.imshow(S, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
                if pi == 0:
                    ax.set_title(col_titles[ci], fontsize=10)
                if ci == 0:
                    ax.set_ylabel(f"{pdb}\nN={S.shape[0]}", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(
            f"Cosine similarity of post-GNN node embeddings ({kind}),"
            f" mean over {len(SEEDS)} seeds",
            fontsize=12,
        )
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="cosine similarity")
        plt.subplots_adjust(left=0.07, right=0.90, top=0.92, bottom=0.05,
                            wspace=0.1, hspace=0.15)
        out_path = os.path.join(FIG_DIR, f"similarity_{kind}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved -> {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
