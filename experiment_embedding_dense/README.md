# Embedding Ablation: FeatureTokenizer vs DenseFeatureEmbedding

This folder contains the ablation study comparing two node/edge feature encoders
used before the message-passing layers of both the Baseline GNN and the GMN.


## How to run the experiment

The whole ablation is driven by a single script in the project root:

```bash
python run_embedding_experiment.py
```

The script runs the full grid `{baseline, gmn} x {tokenizer, dense} x 5 seeds`
(20 runs in total), reusing the hyperparameters defined in `config.yaml`.
Results land in `experiment_embedding_dense/`:

- `runs_detail.csv` -- one row per run (per-seed metrics).
- `runs_summary.csv` -- mean +/- std grouped by (model, embedding).
- `runs/<model>_<embedding>_seed<N>/` -- Lightning logs and `best.ckpt`
  for each run.

The script writes `runs_detail.csv` after every single run, so partial results
survive a crash. A built-in **resume** feature re-reads the existing
`runs_detail.csv` and skips already-completed runs, so you can safely re-launch
the same command to recover failed runs without wasting GPU time.

```


## How to switch between Tokenizer and Dense for a single run

For single training runs (not the ablation), the choice is driven by two keys
in `config.yaml` (or in any custom config you pass to `train.py`):

```yaml
model:
  embedding_type: "tokenizer"   # or "dense"
  cat_emb_dim: 8                # only used when embedding_type == "dense"
```

Then launch:

```bash
python src/train.py config.yaml
```

Internally, `DualGraphTokenizer.__init__` dispatches to `FeatureTokenizer`
or `DenseFeatureEmbedding` according to `embedding_type`. The rest of the
pipeline (GINEConv branches in the baseline, GMN cross-attention layers,
pooling, MLP head) is identical, so the two configurations are directly
comparable.


---


## Dataset pipeline

### 1) `INDEX_general_PL.2020R1.lst`
Raw file downloaded from PDBbind v2020R1, plain-text format (not CSV).
Around 19k rows, one per protein-ligand complex. For each row:
`pdb_code`, resolution, year, binding type (Kd/Ki/IC50), operator (=, <, ~),
value, unit (mM/uM/nM...), reference, ligand name, comments.

### 2) `parsed_index.csv`
Regex-based extraction of the INDEX file into a tabular CSV format.
Same 19k entries, no filtering applied yet. Its only purpose is to turn the
raw PDBbind text into a pandas-friendly table.

### 3) `PL_cleaned_index.csv` (4852 rows)
Result of the filters applied to `parsed_index.csv`. Filters in order:

| # | Filter | Reason |
|---|--------|--------|
| 1 | Remove NMR entries (keep X-ray only) | NMR does not give unique coordinates |
| 2 | Keep only `compare == "="` (drop ~, <, >) | We want exact affinity values, not bounds/approximations |
| 3 | Resolution <= 2.5 Angstrom | Discard low-quality structures |
| 4 | Drop comments with: `incomplete ligand`, `covalent complex`, `covalent bond`, `isomer`, `Different protein/ligand in assay`, `Nonstandard assay` | "Dirty" cases (covalent ligand, non-standard assay, etc.) |
| 5 | Keep only `binding_type == "Kd"` | Deliberate choice: Kd is the most physically meaningful affinity measurement; Ki and IC50 are discarded |
| 6 | Drop `value_M <= 0` | Invalid values |
| 7 | Drop rows without a valid ligand SDF file | Required to compute SMILES and the molecular graph |

The dominant drop is filter #5: keeping only Kd discards about 65% of the
rows (Kd is roughly 35% of PDBbind entries).

### 4) `data/processed/`
Contains PyG graphs precomputed from the cleaned index, two files per
complex: `protein_{pdb}.pt` and `ligand_{pdb}.pt`. Also contains
`cached_schema.json`, which stores the feature schema (number of dimensions,
number of categories for each categorical feature, etc.) for nodes and edges
of both proteins and ligands. This cached schema is consumed by
`DualGraphTokenizer` to align the input dimensions.

The protein/ligand split uses two separate files per complex because the
model treats them as two distinct graphs that are coupled only at training
time (via cross-attention in the GMN, via concatenation at the MLP head in
the baseline).


---


## Node and edge feature embedding

Both the baseline and the GMN start from raw node/edge features and need to
turn them into dense vectors of a common dimension before feeding them into
the message-passing layers. The ablation compares two very different ways
of doing this.

### Baseline strategy: FeatureTokenizer

Each feature is projected independently into a `token_dim`-dimensional
space (`token_dim = 64` with the current config):

- Continuous features: `nn.Linear(1, token_dim)`
- Categorical features: `nn.Embedding(num_categories, token_dim)`

The resulting per-feature tokens are concatenated, producing a vector of
dimension `N * token_dim`:

- Protein node:  9 * 64 = **576**
- Ligand node:   7 * 64 = **448**
- Protein edge:  2 * 64 = **128**
- Ligand edge:   3 * 64 = **192**

These dimensionalities are very different from each other and very different
from the message-passing hidden size. A per-stream MLP then reduces every
vector to a common hidden dimension (16 with the current config), homogeneous
across protein and ligand branches.

### Alternative strategy: DenseFeatureEmbedding (`layer_dense_experiment` branch)

Instead of projecting each feature individually, all features of a given
node (or edge) are treated as a single input vector of size `N` and fed
directly into a Linear layer:

```
Input: [N features]            # e.g. N = 9 for a protein node
  Linear(N, hidden_dim) -> ReLU -> [Linear(hidden_dim, hidden_dim)]
  Output: [hidden_dim]
```

Two subtleties:

1. **All features are treated as one input vector.** Not one at a time, but
   `[meiler_1, ..., meiler_7, b_factor, residue_name]` as a vector of 9
   values. The Linear sees everything at once and produces an embedding.

2. **Categorical features must be preprocessed.** A `Linear(1, ...)` cannot
   handle categorical values such as `residue_name` (0-20) or `hybridization`
   (0-6) directly, because it would interpret them as ordinal numbers:
   "ALA=0 is smaller than LYS=11" is meaningless. Two options:

   - **(a) One-hot encoding.** Transform `residue_name=7` into
     `[0,0,0,0,0,0,0,1,0,...]` of size 20, then concatenate with the
     continuous features. For a protein node this would yield
     `7 (Meiler) + 1 (b-factor) + 20 (residue) = 28` input dimensions.

   - **(b) Embedding + manual concatenation.** Keep an `nn.Embedding` only
     for categorical features, then concatenate those embeddings with the
     raw continuous features before the first Linear.

The current implementation (`features_embedding_dense.py`) uses option (b)
with `cat_emb_dim = 8`.

Input dimensions after preprocessing (with `cat_emb_dim = 8`):

|              | Continuous | Categorical | Input to MLP |
|--------------|-----------:|------------:|-------------:|
| Protein node | 8 * 1      | 1 * 8       | **16**       |
| Ligand node  | 3 * 1      | 4 * 8       | **35**       |
| Protein edge | 1 * 1      | 1 * 8       | **9**        |
| Ligand edge  | 1 * 1      | 2 * 8       | **17**       |

For comparison, with the current `FeatureTokenizer` and `token_dim = 64`,
the MLP received vectors of dimension 576 / 448 / 128 / 192. The Dense
variant reduces the input dimensionality upstream of the MLP by one to two
orders of magnitude, with a proportional reduction in the number of
parameters of the encoder block.


---


## Experiment conducted

Comparison of **tokenizer** vs **dense** on **5 seeds** (42, 47, 51, 53, 59),
on both the Baseline GNN and the GMN. Grid of 20 runs in total. All other
hyperparameters are shared across runs (see `config.yaml`). The experiment
was run on the Politecnico VM (NVIDIA A2, 15 GB) on ~992 preprocessed PDBbind
complexes (train/val split 0.8/0.2 on complex-level PDB codes).

For each run we track:

- **Training**: `best_val_loss`, `best_val_mae`, `best_val_pearson_r`,
  `best_epoch`, `gen_gap` (train loss at best val epoch minus best val loss),
  `smooth_std` (std of consecutive val-loss differences).
- **Oversmoothing**: `MAD` (mean 1 - cosine distance between node pairs
  within each graph), `Dirichlet energy` on graph edges, and the stable
  rank `eff_rank = (sum sigma_i)^2 / sum sigma_i^2` of the post-GNN node
  matrix. All three are computed on protein and ligand branches separately.
- **Feature importance**: norm of the first-Linear weights of each node
  encoder, split by continuous vs categorical input columns
  (`w_norm_cont_*`, `w_norm_cat_*`). A low `w_cont / w_cat` ratio means
  the MLP learned to rely on categorical features and effectively ignore
  the continuous ones.


---


## Results and interpretability

### Aggregated results (mean +/- std over 5 seeds)

|                     | baseline+tok     | baseline+dense     | gmn+tok          | gmn+dense          |
|---------------------|-----------------:|-------------------:|-----------------:|-------------------:|
| params              | 59,553           | **8,329**          | 62,001           | **10,777**         |
| best_epoch          | 158.8 +/- 39.7   | **139.4** +/- 41.7 | 159.0 +/- 47.2   | **138.4** +/- 20.9 |
| best_val_loss       | 2.580 +/- 0.343  | 2.560 +/- **0.187**| **2.493** +/- 0.277 | 2.549 +/- **0.224** |
| best_val_mae        | 1.276 +/- 0.068  | 1.278 +/- **0.043**| **1.254** +/- 0.067 | 1.273 +/- **0.030** |
| best_val_r          | 0.565 +/- 0.051  | 0.566 +/- 0.063    | **0.582** +/- 0.060 | 0.572 +/- **0.040** |
| gen_gap             | **0.196**        | 0.381              | 0.549            | **0.355**          |
| mad_prot            | 0.076            | **0.233**          | 0.059            | **0.129**          |
| eff_rank_prot       | 2.045            | **3.039**          | 2.290            | **3.228**          |
| eff_rank_lig        | 3.57             | **4.24**           | 6.59             | **6.93**           |
| w_cont_prot / w_cat_prot | **1.18**    | 0.74               | **1.11**         | 0.68               |
| w_cont_lig / w_cat_lig   | **0.86**    | 0.48               | **0.87**         | 0.44               |

### Interpretation

**Parameter count and convergence speed.** Dense has roughly 7x fewer
parameters than Tokenizer and converges about 20 epochs earlier in both
model families. Faster convergence here does not imply a better model: it
reflects the lower capacity of Dense, which saturates earlier on its own
(lower) performance ceiling.

**Task-level metrics.** The gap between the two encoders on the final
validation loss is small and lies within the per-seed standard deviation
in both model families, i.e. the difference is not statistically
significant. The direction is however informative:

- On the **Baseline**, Dense and Tokenizer are essentially equivalent
  (2.560 vs 2.580 on `best_val_loss`).
- On the **GMN**, Tokenizer is consistently better than Dense
  (2.493 vs 2.549 on `best_val_loss`; 0.582 vs 0.572 on Pearson R).

This asymmetry is consistent with the architectural role of the two models.
In the Baseline, the two GNN branches do not communicate until the final
MLP, so the quality of individual node embeddings is less critical; the
over-parameterised Tokenizer brings no benefit over Dense. In the GMN,
cross-attention compares nodes one-to-one at every layer: the richer the
per-node embedding, the more discriminative the attention maps become.
Dense, being poorer per-node, penalises the GMN more.

**Stability across seeds.** Dense is consistently more stable than
Tokenizer: `std(best_val_loss)` drops by 45% on the Baseline (0.343 ->
0.187) and by 19% on the GMN (0.277 -> 0.224); `std(best_val_mae)` drops
by 37% and 55% respectively. Fewer parameters and fewer redundant degrees
of freedom result in more similar converged solutions across different
random seeds.

**Oversmoothing.** The three oversmoothing proxies agree on a somewhat
counter-intuitive conclusion: **Dense suffers less from oversmoothing
than Tokenizer**.

- `eff_rank_prot` increases by 40-48% with Dense in both model families
  (2.0 -> 3.0 in the baseline, 2.3 -> 3.2 in the GMN).
- `mad_prot` is roughly 2-3x higher with Dense.
- `dirichlet` is extremely noisy (its std is of the same order as its
  mean), so it is not a reliable signal here.

The mechanism is the following. Tokenizer first expands the raw features
into a very wide intermediate space (9 x 64 = 576 dimensions for a protein
node) and then compresses them to the hidden size of the message-passing
layers (16-32 dimensions), i.e. an 18:1 compression. The MLP learns the
dominant linear combinations and discards variation along the remaining
directions. Dense instead starts from a small input (16 dimensions for a
protein node) and expands it to 32: no information is explicitly thrown
away upstream, and the MLP has room to preserve per-node variability.

**Feature importance.** Despite being less oversmoothed, Dense pays a
real price on the semantic content of the embedding. The ratio
`w_norm_cont / w_norm_cat` on the first Linear of each node encoder drops
systematically with Dense:

- Protein nodes: ratio goes from 1.1-1.2 (Tokenizer) to 0.7 (Dense).
- Ligand nodes: ratio goes from 0.87 (Tokenizer) to 0.44-0.48 (Dense).

With Dense, the MLP relies disproportionately on categorical features
and effectively down-weights the continuous ones (Meiler descriptors,
b-factor, bond and 3D distances). The node variability is still high
(categoricals have many levels: 21 residues, 120 atomic numbers, etc.),
which is why `eff_rank` and `MAD` stay high. But the fine-grained
physico-chemical information carried by continuous features is partially
lost. In the GMN, where cross-attention needs to match protein pockets
to ligand substructures, this loss of chemical granularity is a plausible
explanation for the observed drop in `best_val_loss` and `best_val_r`.

### Takeaway

- Dense is a competitive, more compact and more stable alternative to
  Tokenizer on the Baseline (where per-node expressivity is not critical).
- Tokenizer should be preferred for the GMN, where per-layer cross-attention
  benefits from richer per-node embeddings that preserve the continuous
  (physico-chemical) features.
- Caveat: the current experiment is run on ~992 complexes, with per-seed
  standard deviations of similar magnitude as the between-encoder
  differences. A larger run on the full PDBbind refined set is recommended
  before drawing stronger conclusions, especially on the GMN direction.


---


## Visualization: cosine-similarity heatmaps

To inspect oversmoothing qualitatively, we produce cosine-similarity heatmaps
of the post-GNN node embeddings. The script `plot_similarity_heatmaps.py`
generates two figures in `experiment_embedding_dense/figures/`:

- `similarity_proteins.png`: 3 protein graphs x 4 model-embedding configurations.
- `similarity_ligands.png`:  3 ligand graphs  x 4 model-embedding configurations.

The 3 proteins and 3 ligands are picked automatically from the preprocessed
complexes (same PDB code for both figures). The 4 columns in each figure are
`baseline+tokenizer`, `baseline+dense`, `gmn+tokenizer`, `gmn+dense`.

### What a cell represents

Each heatmap is an `N x N` matrix where `N` is the number of nodes of that
graph (residues for proteins, atoms for ligands). The cell at position
`(i, j)` is the cosine similarity between the post-GNN embedding of node
`i` and the post-GNN embedding of node `j`:

```
S[i, j] = (X[i] . X[j]) / (||X[i]|| . ||X[j]||)
```

where `X[i] in R^16` is the 16-dimensional embedding of node `i` produced
by the GNN (with `num_layers = 1`, this is the output of the single
message-passing layer: `tokenizer -> GINEConv` for the baseline,
`tokenizer -> GMNLayer` for the GMN). Edge embeddings are NOT shown:
oversmoothing is a node-level phenomenon, and in the current model
edge features are not updated by the message-passing layers.

### Matrix properties

- **Diagonal is always 1** (`S[i, i] = 1`): a vector is perfectly similar
  to itself. Read the off-diagonal, not the diagonal.
- **Symmetric**: `S[i, j] = S[j, i]`, the matrix is mirrored across the
  diagonal.
- **Range `[-1, 1]`** in theory. In practice, post-ReLU embeddings live
  in a positive cone, so almost all cosines are `>= 0`; negative cells
  (blue) are rare and indicate genuinely opposite directions in the
  latent space.

### Colormap and reading guide

Colormap `RdBu_r`, fixed range `[-1, 1]`:

| cosine | colour       | geometric meaning        | interpretation                                             |
|-------:|:-------------|:-------------------------|:-----------------------------------------------------------|
| **+1** | dark **red** | same direction           | nodes **very similar** (collapsed -> oversmoothing)        |
| 0      | **white**    | orthogonal vectors       | nodes **well differentiated** (what we want)               |
| **-1** | dark **blue**| opposite directions      | nodes **anti-correlated** (rare, a sign of strong contrast)|

So: **uniformly red matrix = heavy oversmoothing; contrasted matrix
(white and red patches) = well-differentiated node embeddings**.

### How the matrices are obtained

For each of the 4 configurations, the script loads all 5 seed checkpoints
and, for each selected graph, runs a forward pass per seed (5 runs x 3
graphs x 4 configurations = 60 forward passes in total). It captures the
output of the last message-passing layer via a forward hook, computes
`S = X_norm @ X_norm.T`, and averages the 5 resulting matrices
element-wise. The averaging is well-defined because the node order is
fixed by the stored `.pt` graph file and is identical across seeds.

### Interpretation of the figures

The heatmaps confirm the quantitative story told by the aggregated MAD
and effective-rank metrics, but add important nuances.

**Proteins.** On a mid-sized protein (e.g. `184l`, N = 162 residues) the
tokenizer/dense difference is clearly visible: the tokenizer produces
a nearly uniform deep-red matrix (all residues collapsed), while dense
shows a lighter background with a visible sub-domain checkerboard,
indicating that residues are still distinguishable after one
message-passing step. On large proteins (e.g. `1mwt` with N = 1261 or
`2e7l` with N = 814), all four configurations saturate to uniform deep
red: with `num_layers = 1`, a message reaches only direct neighbours and
the mean pool compresses hundreds of nodes into 16 dimensions, forcing
high correlations regardless of the encoder. The tokenizer-vs-dense
difference on large proteins is real (it shows up in the aggregated
`mad_prot` and `eff_rank_prot` numbers in the tables above) but is
drowned by the saturated colormap in this visualization.

**Ligands.** On ligands the dominant axis of variation is **not**
tokenizer vs dense but **baseline vs GMN**. The baseline branches (both
with tokenizer and with dense) produce visibly contrasted matrices, and
on ligands `1mwt` (N = 24) and `2e7l` (N = 76) they even reach slightly
**negative cosines** (pale-blue cells) -- atoms in anti-correlated
directions in the latent space, which is a strong signal of successful
differentiation. The GMN columns stay mostly red: per-layer
cross-attention with the protein tends to average ligand atoms towards
protein-informed directions, smoothing their embeddings. On ligands the
dense/tokenizer choice is roughly equivalent, as expected from the fact
that ligand features are dominated by categoricals (atomic number,
aromaticity, in-ring, hybridization), which dense handles well.

**Operational summary.**
- Dense reduces oversmoothing on protein graphs when the graph is not
  too large to begin with.
- The oversmoothing penalty on ligands is architectural (introduced by
  GMN cross-attention), not driven by the encoder choice.
- For a cleaner visual comparison on proteins, replacing large targets
  (> 500 residues) with medium-sized ones (100-300 residues) makes the
  tokenizer/dense gap much more evident in the heatmaps.
