# GMN-binding

Protein–ligand binding affinity prediction (pAffinity = −log10 Kd/Ki) with a **Graph Matching Network**:
protein and ligand graphs exchange information through cross-graph attention during message passing.
A baseline with two independent GNNs (same tokenizer, pooling and prediction head) is used as a controlled comparison.

- Protein: residue-level graph (Graphein), peptide bonds + Cα–Cα contacts within 8 Å
- Ligand: atom-level graph (RDKit) from the crystal pose, chemical bonds + atom pairs within 5 Å
- Data: PDBbind v2020R1, test on the CASF-2016 core set

## Repository structure

```
gmn-binding/
├── configs/
│   ├── gmn.yaml                        experiment config: data version, split, model, training (GMN)
│   ├── baseline.yaml                   same for the baseline (two independent GNNs)
│   └── data/
│       ├── complex_features.yaml       featurization config: version, protein/ligand graph parameters
│       ├── index_v2020_generalset.yaml index filters used for the current dataset (Kd + Ki, ≤ 2.75 Å)
│       └── index_v2020.yaml            index filters of the exam/LIRE dataset (Kd only, ≤ 2.5 Å)
│
├── data/
│   ├── raw/                            PDBbind files, see "Data" (not versioned)
│   ├── index/                          (versioned)
│   │   ├── index.csv                   current index: pdb_code, resolution, pAffinity, SMILES
│   │   ├── parsed_index.csv            full parsed INDEX_general_PL.2020R1 (19,037 complexes, no filters)
│   │   ├── PL_cleaned_index.csv        index of the exam/LIRE dataset (Kd only, 4,852 complexes)
│   │   ├── codes_core.txt              CASF-2016 core set (285 PDB codes)
│   │   ├── codes_refined.txt           PDBbind v2020 refined set (validation candidates)
│   │   ├── codes_general.txt           PDB codes of index.csv
│   │   └── splits/
│   │       ├── v2020_casf/             train.txt, val.txt, test.txt (one PDB code per line) + split_info.json
│   │       └── v2020_casf_cv5/         fold_0 … fold_4 (train.txt, val.txt) + cv_info.json
│   └── processed/                      graphs (.pt files not versioned, meta.json versioned)
│       ├── v1/                         exam/LIRE dataset (Kd only, 4,851 complexes)
│       └── v2/                         current dataset (9,560 complexes)
│           ├── protein_<code>.pt       protein graph (PyG Data: x, edge_index, edge_attr, pos, y)
│           ├── ligand_<code>.pt        ligand graph
│           └── meta.json               config, git commit, counts and excluded complexes
│
├── scripts/                            command-line steps (run from the repo root)
│   ├── build_index.py                  official INDEX file -> filtered index with pAffinity and SMILES
│   ├── extract_codes.py                PDB codes of a PDBbind INDEX file -> .txt list
│   ├── make_splits.py                  train/val/test split (test = CASF-2016 core set)
│   ├── make_cv_splits.py               K-fold CV splits over train + val of a split
│   ├── preprocess.py                   protein and ligand graphs -> data/processed/<version>/
│   └── train.py                        trains one model from an experiment config
│
├── src/gmnbind/                        Python package (pip install -e .)
│   ├── data/
│   │   ├── protein.py                  Protein: residue-level graph (Graphein)
│   │   ├── ligand.py                   Ligand: atom-level graph (RDKit)
│   │   ├── schemas.py                  type of every node/edge feature (continuous or categorical)
│   │   ├── pdbbind_dataset.py          builds and saves the .pt graphs (used by preprocess.py)
│   │   ├── complex_dataset.py          loads (protein, ligand) pairs + collate function
│   │   ├── datamodule.py               Lightning DataModule: reads a split and loads its complexes
│   │   └── splits.py                   read/write split files
│   ├── models/
│   │   ├── tokenizer.py                per-feature embedding of nodes and edges
│   │   ├── layers.py                   CrossGraphAttention and GMNLayer
│   │   ├── base.py                     shared LightningModule: loss, metrics, optimizer, prediction head
│   │   ├── gmn.py                      GraphMatchingNetwork
│   │   └── baseline.py                 BaselineGNN
│   ├── metrics.py                      RMSE, MAE, Pearson, Spearman, SD, CI
│   ├── training.py                     train(): one training run from a config
│   └── cross_validation.py             cross_validate(): train() on every fold + summary
│
├── notebooks/
│   ├── experiments.ipynb               cross-validation of several configurations + comparison plots
│   ├── attention_graph.ipynb           visualization of the cross-graph attention
│   ├── protein_overview.ipynb          exploration of the protein graphs
│   ├── ligand_overview.ipynb           exploration of the ligand graphs
│   ├── PDB_index_processing.ipynb      exploration of the PDBbind index
│   └── chembl_dataset_overview.ipynb   exploration of ChEMBL data
│
├── lightning_logs/                     training outputs (not versioned)
│   ├── <experiment>/version_N/         train.py: metrics.csv, hparams.yaml, best.ckpt
│   ├── <experiment>_cv/fold_i/         cross-validation: metrics.csv, hparams.yaml
│   │   ├── cv_folds.csv                one row per fold (best epoch + val metrics)
│   │   └── cv_summary.json             mean ± std of every metric
│   └── summary.csv                     one row per train.py run
│
├── pyproject.toml                      package definition
├── requirements.txt                    dependencies
├── REFACTOR.md                         refactor status and known issues
├── .gitattributes                      LF line endings + nbstripout for notebooks
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash); Linux/macOS: source .venv/bin/activate

# GPU: install torch from the PyTorch index first (pick the CUDA version you have)
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install -e .                     # always with -e
nbstripout --install                 # strips notebook outputs before commits (once per clone)
```

## Data

Download from [PDBbind+](https://www.pdbbind-plus.org.cn/) (registration required) and place them as:

```
data/raw/pdbbind_v2020/
├── INDEX_general_PL.2020R1.lst
├── INDEX_refined_set.2020
├── INDEX_core_set.2016.dat          # CASF-2016 core set
└── generalset/<pdb_code>/<pdb_code>_protein.pdb | _pocket.pdb | _ligand.sdf
```

## Pipeline

Each step reads a config or the constants at the top of the script, and writes to `data/`.

| Step | Command | Output |
|---|---|---|
| 1. Index | `python scripts/build_index.py --config configs/data/index_v2020_generalset.yaml` | `data/index/index.csv` |
| 2. Code lists | `python scripts/extract_codes.py <INDEX file> <output.txt> [--keep-in <csv>]` | `data/index/codes_core.txt`, `codes_refined.txt` |
| 3. Splits | `python scripts/make_splits.py` · `python scripts/make_cv_splits.py` | `data/index/splits/` |
| 4. Graphs | `python scripts/preprocess.py` | `data/processed/<version>/` |
| 5. Train | `python scripts/train.py --config configs/gmn.yaml` | `lightning_logs/<experiment>/version_N/` |
| 6. Cross-validation | `python -m gmnbind.cross_validation configs/gmn.yaml` | `lightning_logs/<experiment>_cv/` |

Steps 1–3 are already done for the versioned files in `data/index/`: after downloading the raw data, start from step 4.

**Current dataset** (Kd + Ki, exact values, resolution ≤ 2.75 Å, NMR and problematic entries excluded): 9,561 complexes.

| Split `v2020_casf` | Complexes |
|---|---|
| train | 7,885 |
| val | 1,391 (15% of train+val, sampled from the refined set) |
| test | 285 (CASF-2016 core set) |

`v2020_casf_cv5`: 5 folds over train + val (the test set is never used in the CV).

## Data versions

The graphs depend on `configs/data/complex_features.yaml` (edges, cutoffs, protein vs pocket, …) and on the
featurization code (`protein.py`, `ligand.py`, `pdbbind_dataset.py`). They are saved in `data/processed/<version>/`,
where `<version>` is the `version` field of the config.

- `meta.json` records the config, the git commit, the counts and the excluded complexes.
- `preprocess.py` resumes an interrupted run, and stops if the config differs from the one in `meta.json`.
- **If you change the config or the featurization code, bump `version`.**

## Training and evaluation

- The experiment config (`configs/gmn.yaml`, `configs/baseline.yaml`) sets data version, split, model and training.
  `model.model_type` selects the model (`gmn` or `baseline`).
- Early stopping on `val_loss`; the result of a run is the epoch with the lowest `val_loss`.
- Metrics on the whole validation/test set: RMSE, MAE, Pearson, Spearman, SD (CASF definition) and CI (`gmnbind.metrics`).
- `train.py` saves `best.ckpt` and appends a row to `lightning_logs/summary.csv`.
  The cross-validation saves no checkpoint, but writes `cv_folds.csv` (one row per fold) and `cv_summary.json` (mean ± std).
- `notebooks/experiments.ipynb` runs and compares several configurations with the cross-validation.
