"""
Cosine-similarity heatmaps of post-GNN node embeddings, for the v2 dataset.

Refactored-pipeline version of the original plot_similarity_heatmaps.py
(layer_dense_experiment branch), pointed at the v2 runs and graphs.

For each of 4 model/embedding configurations ({baseline, gmn} x {tokenizer, dense}),
loads the seed checkpoint(s) in experiment_embedding_dense/runs/v2/, runs inference on
3 protein-ligand complexes from data/processed/v2/, extracts the node embeddings AFTER
the last message-passing layer, and plots the cosine-similarity matrix
S = X_norm @ X_norm.T (averaged over the available seeds).

Interpretation: more uniform/red heatmap = nodes collapsed together (stronger
oversmoothing); more contrasted heatmap = nodes more differentiated.

Outputs:
  experiment_embedding_dense/figures/similarity_proteins_v2.png
  experiment_embedding_dense/figures/similarity_ligands_v2.png

Usage (on the VM, from the repo root):
    python scripts/plot_similarity_heatmaps.py

To compare directly with the v1 figure, set PDB_CODES below to the same 3 codes
shown on the left of the v1 image (experiment_embedding_dense/figures/similarity_proteins.png),
as long as they also exist in data/processed/v2/.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))   # make gmnbind importable without `pip install -e .`

from gmnbind.models.baseline import BaselineGNN
from gmnbind.models.gmn import GraphMatchingNetwork

DATASET_TAG = "v2"
RUNS_DIR  = REPO_ROOT / "experiment_embedding_dense" / "runs" / DATASET_TAG
FIG_DIR   = REPO_ROOT / "experiment_embedding_dense" / "figures"
PROCESSED = REPO_ROOT / "data" / "processed" / DATASET_TAG
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42]   # v2 was trained on a single seed; averaging collapses to that seed
CONFIGS = [
    ("baseline", "tokenizer"),
    ("baseline", "dense"),
    ("gmn",      "tokenizer"),
    ("gmn",      "dense"),
]

# PDB codes to plot. None -> picked automatically (3 complexes spread across the
# sorted list). Set to a list of 3 codes to force a choice, e.g.
# PDB_CODES = ["1a1e", "1b40", "2zff"]   (use the same 3 as the v1 figure to compare)
PDB_CODES = None

MODEL_CLS = {"baseline": BaselineGNN, "gmn": GraphMatchingNetwork}


# -------------------- Helpers --------------------
def pick_pdb_codes(n=3):
    """Pick n PDB codes that have BOTH protein_<pdb>.pt and ligand_<pdb>.pt."""
    prot = {Path(p).name[len("protein_"):-len(".pt")] for p in glob.glob(str(PROCESSED / "protein_*.pt"))}
    lig  = {Path(p).name[len("ligand_"):-len(".pt")]  for p in glob.glob(str(PROCESSED / "ligand_*.pt"))}
    common = sorted(prot & lig)
    if len(common) < n:
        raise RuntimeError(f"Only {len(common)} complexes available in {PROCESSED}; need {n}.")
    step = max(1, len(common) // n)   # spread the picks across the sorted list
    return common[::step][:n]


def load_complex(pdb):
    """Load a single complex as a 1-graph batch (protein and ligand)."""
    prot = torch.load(PROCESSED / f"protein_{pdb}.pt", weights_only=False)
    lig  = torch.load(PROCESSED / f"ligand_{pdb}.pt",  weights_only=False)
    return Batch.from_data_list([prot]), Batch.from_data_list([lig])


@torch.no_grad()
def extract_last_embeddings(model, prot_batch, lig_batch, model_type):
    """Forward hook on the last message-passing layer(s) -> post-GNN node embeddings."""
    device = next(model.parameters()).device
    prot_batch = prot_batch.to(device)
    lig_batch = lig_batch.to(device)
    captured = {}

    if model_type == "baseline":
        h1 = model.prot_gnn_layers[-1].register_forward_hook(
            lambda m, i, o: captured.setdefault("prot", o.detach()))
        h2 = model.lig_gnn_layers[-1].register_forward_hook(
            lambda m, i, o: captured.setdefault("lig", o.detach()))
        hooks = [h1, h2]
    else:
        h = model.gmn_layers[-1].register_forward_hook(
            lambda m, i, o: captured.setdefault("gmn_out", o))
        hooks = [h]

    model.eval()
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
    """Load the best checkpoint for a (model, embedding, seed) run, or None if missing."""
    ckpt_path = RUNS_DIR / f"{model_type}_{embedding_type}_seed{seed}" / "version_0" / "best.ckpt"
    if not ckpt_path.exists():
        return None
    return MODEL_CLS[model_type].load_from_checkpoint(str(ckpt_path), map_location="cpu")


# -------------------- Main --------------------
def main():
    pdb_codes = PDB_CODES or pick_pdb_codes(3)
    # keep only codes whose protein AND ligand .pt both exist in v2 (skip the rest)
    existing = [c for c in pdb_codes
                if (PROCESSED / f"protein_{c}.pt").exists() and (PROCESSED / f"ligand_{c}.pt").exists()]
    missing = [c for c in pdb_codes if c not in existing]
    if missing:
        print(f"WARNING: {missing} not found in {PROCESSED}, skipped")
    pdb_codes = existing
    if not pdb_codes:
        raise RuntimeError(f"None of the requested PDB codes exist in {PROCESSED}.")
    print(f"Using PDB codes: {pdb_codes}")

    # sim_mats[config_idx][pdb_idx] -> (S_prot_mean, S_lig_mean)
    sim_mats = [[None] * len(pdb_codes) for _ in CONFIGS]

    for ci, (model_type, embedding_type) in enumerate(CONFIGS):
        print(f"\n=== {model_type} / {embedding_type} ===")
        models = []
        for s in SEEDS:
            m = load_ckpt(model_type, embedding_type, s)
            if m is None:
                print(f"  [skip] no checkpoint for seed {s}")
                continue
            models.append(m)
        if not models:
            print(f"  [FAIL] no checkpoints for {model_type}/{embedding_type} -> empty column")
            continue

        for pi, pdb in enumerate(pdb_codes):
            prot_batch, lig_batch = load_complex(pdb)
            S_prot_list, S_lig_list = [], []
            for m in models:
                x_p, x_l = extract_last_embeddings(m, prot_batch, lig_batch, model_type)
                S_prot_list.append(cosine_sim_matrix(x_p))
                S_lig_list.append(cosine_sim_matrix(x_l))
            sim_mats[ci][pi] = (
                np.mean(np.stack(S_prot_list), axis=0),
                np.mean(np.stack(S_lig_list), axis=0),
            )
            print(f"  {pdb}: prot {sim_mats[ci][pi][0].shape}, lig {sim_mats[ci][pi][1].shape}")

    # -------------------- Plot --------------------
    seed_tag = f"seed {SEEDS[0]}" if len(SEEDS) == 1 else f"mean over {len(SEEDS)} seeds"
    col_titles = [f"{mt}\n{et}" for mt, et in CONFIGS]
    for kind, idx in [("proteins", 0), ("ligands", 1)]:
        fig, axes = plt.subplots(
            len(pdb_codes), len(CONFIGS),
            figsize=(3.2 * len(CONFIGS), 3.2 * len(pdb_codes)),
            squeeze=False,
        )
        im = None
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
            f"Cosine similarity of post-GNN node embeddings ({kind}, {DATASET_TAG}), {seed_tag}",
            fontsize=12,
        )
        if im is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            fig.colorbar(im, cax=cbar_ax, label="cosine similarity")
        plt.subplots_adjust(left=0.07, right=0.90, top=0.92, bottom=0.05, wspace=0.1, hspace=0.15)
        out_path = FIG_DIR / f"similarity_{kind}_{DATASET_TAG}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved -> {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
