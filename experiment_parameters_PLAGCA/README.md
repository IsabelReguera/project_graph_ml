# Experiment: PLAGCA Hyperparameters and Graph Density

Goal: assess how far the current model (encoder = Dense) can go with (i) the
hyperparameters from a published paper on the same task (PLAGCA, Shi et al.
2025) and (ii) a sparser graph construction (tighter protein distance
threshold, no spatial edges on the ligand).

All runs use `DenseFeatureEmbedding` as the node/edge encoder. The dataset,
preprocessed graphs at 8 Angstrom (protein) and 5 Angstrom + chemical bonds
(ligand), remain unchanged; any graph-density modifications are applied
on-the-fly during data loading.


## Configurations

Three configurations are compared. Only the parameters that differ from
configuration A are listed in B and C.

### Model architecture

| Parameter             | A (current) | B (PLAGCA) | C (PLAGCA + sparse graph) |
|-----------------------|-------------|------------|---------------------------|
| encoder               | Dense       | Dense      | Dense                     |
| cat_emb_dim           | 8           | 8          | 8                         |
| node_embedding_dims   | [32, 16]    | [128]      | [128]                     |
| edge_embedding_dims   | [32, 16]    | [128]      | [128]                     |
| num_layers (T)        | 1           | 3          | 3                         |
| MLP head sizes        | 2*emb, emb, 1 | 512, 256, 128, 1 | 512, 256, 128, 1 |
| MLP dropout           | 0.12        | 0.5        | 0.5                       |
| activation            | ReLU        | ReLU       | ReLU                      |

### Training

| Parameter   | A (current)             | B (PLAGCA)   | C (PLAGCA + sparse graph) |
|-------------|-------------------------|--------------|---------------------------|
| optimizer   | AdamW                   | Adam         | Adam                      |
| scheduler   | warmup + cosine         | none         | none                      |
| lr          | 0.0006                  | 0.001        | 0.001                     |
| batch_size  | 32                      | 512          | 512                       |
| epochs      | 1000 (early stop 30)    | 500          | 500                       |

### Graph edges

| Graph    | A (current)              | B (PLAGCA)               | C (PLAGCA + sparse graph) |
|----------|--------------------------|--------------------------|---------------------------|
| protein  | distance threshold 8 A   | distance threshold 8 A   | distance threshold 4 A    |
| ligand   | chemical bonds + 5 A     | chemical bonds + 5 A     | chemical bonds only       |

Reference for the PLAGCA column: Shi et al., "PLAGCA: Predicting protein-ligand
binding affinity with the graph cross-attention mechanism",
*Journal of Biomedical Informatics* 165 (2025).


## Models

Both model families are evaluated in every configuration:

- `BaselineGNN` (two independent GINEConv branches, concat + MLP head)
- `GraphMatchingNetwork` (per-layer cross-attention between branches)


## Seeds

Each `(config, model)` pair is repeated with 5 seeds for statistical
reliability: **`{42, 47, 51, 53, 59}`**.

Total number of runs:
`3 configurations x 2 models x 5 seeds = 30 runs`.


## Batch size note

All configurations that use PLAGCA hyperparameters (B and C) are run with
**physical batch size 512** (no gradient accumulation) on the Politecnico VM
(NVIDIA A2, 15 GB VRAM). If OOM occurs on large protein graphs, we will
reconsider switching to gradient accumulation.


## Metrics tracked

Reused from the previous embedding-ablation experiment:

- Training/task: `best_val_loss`, `best_val_mae`, `best_val_pearson_r`,
  `best_epoch`, `gen_gap`, `smooth_std`, `params`.
- Oversmoothing (post-GNN on a validation batch): `mad_prot/lig`,
  `dirichlet_prot/lig`, `eff_rank_prot/lig`.
- Feature importance (first-Linear of each node encoder, split by input
  column groups): `w_norm_cont_prot/lig`, `w_norm_cat_prot/lig`.
