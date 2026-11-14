import torch
import torch.nn as nn
from gmnbind.data.schemas import DATASET_SCHEMAS

PROTEIN_NODE_SCHEMA = DATASET_SCHEMAS['protein_node']
LIGAND_NODE_SCHEMA = DATASET_SCHEMAS['ligand_node']
PROTEIN_EDGE_SCHEMA = DATASET_SCHEMAS['protein_edge']
LIGAND_EDGE_SCHEMA = DATASET_SCHEMAS['ligand_edge']


class FeatureTokenizer(nn.Module):
    """
    Projects each feature individually into a token_dim space,
    concatenates the tokens and aggregates them with a variable-depth MLP.

    Continuous features:   Linear(1, token_dim)
    Categorical features:  nn.Embedding(num_categories, token_dim)

    Aggregation: Linear(N·token_dim, hidden_dim) -> ReLU -> ... -> Linear(..., hidden_dim[-1])
    Output: [N_nodes, hidden_dim[-1]]
    """

    def __init__(self, schema, token_dim, mlp_hidden_layers=None):
        """
        Args:
          schema: list of dicts {'type': 'continuous'|'categorical', ['num_categories': int]}
          token_dim: projection dimension for each individual feature.
          mlp_hidden_layers: list of ints for the MLP aggregator layer sizes (optional).
            If None, forward returns the raw token concatenation [N, n_features * token_dim].
        """
        super().__init__()
        self.schema = schema
        n=len(schema)

        projectors = []
        for feat in schema:
            if feat['type'] == 'continuous':
                projectors.append(nn.Linear(1, token_dim))
            elif feat['type'] == 'categorical':
                projectors.append(nn.Embedding(feat['num_categories'], token_dim))
            else:
                raise ValueError(f"Tipo di feature sconosciuto: {feat['type']}")
        self.projectors = nn.ModuleList(projectors)

        if mlp_hidden_layers is not None:
            mlp_input_dim = n * token_dim
            n_layers = len(mlp_hidden_layers)
            layers = [nn.Linear(mlp_input_dim, mlp_hidden_layers[0])]
            for i in range(1, n_layers):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(mlp_hidden_layers[i - 1], mlp_hidden_layers[i]))
            self.mlp = nn.Sequential(*layers)
            
        else:
            self.mlp = None
    
    def forward(self, x):
        """
        Args:
          x: [N, n_features] float tensor. Categorical columns are float
             but are cast to long internally.
        Returns:
          [N, mlp_hidden_layers[-1]] if MLP is present, else [N, n_features * token_dim].
        """
        tokens = []
        for i, (feat, proj) in enumerate(zip(self.schema, self.projectors)):
            if feat['type'] == 'categorical':
                tokens.append(proj(x[:, i].long()))
            else:
                tokens.append(proj(x[:, i:i + 1]))

        out = torch.cat(tokens, dim=-1)
        return self.mlp(out) if self.mlp is not None else out


class DenseFeatureEmbedding(nn.Module):
    """
    Alternative to FeatureTokenizer, selected in DualGraphTokenizer via
    embedding_type="dense".

    Instead of projecting each feature individually to token_dim and then
    aggregating them with an MLP, this module:
      1) Embeds every categorical feature with a small nn.Embedding
         (dimension `cat_emb_dim`).
      2) Keeps every continuous feature as a scalar.
      3) Concatenates the resulting vector and passes it through a single
         feed-forward MLP that projects it to the final hidden dimension.

    Output shape matches FeatureTokenizer, so the two encoders are
    interchangeable inside DualGraphTokenizer.
    """

    def __init__(self, schema, hidden_sizes, cat_emb_dim=8):
        """
        Args:
          schema: list of dicts {'type': 'continuous' | 'categorical',
                                 ['num_categories': int]}
          hidden_sizes: list of ints for the MLP (same convention as
                        FeatureTokenizer; last element is the output dim).
          cat_emb_dim: embedding size used for every categorical feature.
        """
        super().__init__()
        self.schema = schema

        # Per-feature preprocessing: one Embedding per categorical feature,
        # continuous features pass through as scalars.
        self.cat_embeddings = nn.ModuleDict()
        input_dim = 0
        for i, feat in enumerate(schema):
            if feat['type'] == 'categorical':
                # Keys are stringified indices (ModuleDict requires string keys).
                self.cat_embeddings[str(i)] = nn.Embedding(feat['num_categories'], cat_emb_dim)
                input_dim += cat_emb_dim
            elif feat['type'] == 'continuous':
                input_dim += 1
            else:
                raise ValueError(f"Unknown feature type: {feat['type']}")

        # Final MLP that mixes all features together and projects to hidden_sizes[-1].
        layers = [nn.Linear(input_dim, hidden_sizes[0])]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
          x: [N, n_features] float tensor. Categorical columns are stored as
             floats but cast to long before being passed to nn.Embedding.
        Returns:
          [N, hidden_sizes[-1]] tensor.
        """
        parts = []
        for i, feat in enumerate(self.schema):
            if feat['type'] == 'categorical':
                # Lookup embedding for each categorical index.
                parts.append(self.cat_embeddings[str(i)](x[:, i].long()))
            else:
                # Keep continuous feature as-is (shape [N, 1]).
                parts.append(x[:, i:i + 1])

        # Concatenate [continuous scalars + categorical embeddings] along feature dim,
        # then project to final hidden space.
        concat = torch.cat(parts, dim=-1)
        return self.mlp(concat)


class DualGraphTokenizer(nn.Module):
    """
    Tokenizes nodes and edges of both protein and ligand graphs into the same
    latent space using one FeatureTokenizer per tensor.
    Receives and returns PyG Data/Batch objects with updated .x and .edge_attr.
    """

    def __init__(self, token_dim,
                 node_hidden_sizes, edge_protein_hidden_sizes, edge_ligand_hidden_sizes,
                 embedding_type="tokenizer", cat_emb_dim=8):
        """
        Args:
          token_dim: individual projection dimension for each feature.
            The last element is the final protein node embedding dimension.
          node_hidden_sizes: list of ints for the single node MLP.
          edge_protein_hidden_sizes: list of ints for the protein edge MLP.
          edge_ligand_hidden_sizes: list of ints for the ligand edge MLP.
          embedding_type: feature encoder to use, "tokenizer" (FeatureTokenizer)
            or "dense" (DenseFeatureEmbedding). The two are interchangeable.
          cat_emb_dim: categorical embedding size, used only when embedding_type="dense".
        """
        super().__init__()

        self.norm_prot_nodes = nn.LayerNorm(node_hidden_sizes[-1])
        self.norm_lig_nodes = nn.LayerNorm(node_hidden_sizes[-1])
        self.norm_prot_edges = nn.LayerNorm(edge_protein_hidden_sizes[-1])
        self.norm_lig_edges  = nn.LayerNorm(edge_ligand_hidden_sizes[-1])

        if embedding_type == "tokenizer":
            def make_enc(schema, hidden_sizes):
                return FeatureTokenizer(schema, token_dim, mlp_hidden_layers=hidden_sizes)
        elif embedding_type == "dense":
            def make_enc(schema, hidden_sizes):
                return DenseFeatureEmbedding(schema, hidden_sizes=hidden_sizes, cat_emb_dim=cat_emb_dim)
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")

        self.protein_node_enc = make_enc(PROTEIN_NODE_SCHEMA, node_hidden_sizes)
        self.protein_edge_enc = make_enc(PROTEIN_EDGE_SCHEMA, edge_protein_hidden_sizes)
        self.ligand_node_enc  = make_enc(LIGAND_NODE_SCHEMA,  node_hidden_sizes)
        self.ligand_edge_enc  = make_enc(LIGAND_EDGE_SCHEMA,  edge_ligand_hidden_sizes)

    def forward(self, prot_batch_in, lig_batch_in):
        """
        Args:
          prot_batch_in, lig_batch_in: PyG Data or Batch with .x and .edge_attr.
        Returns:
          prot_batch, lig_batch: copies with .x [N, node_dim] and .edge_attr [E, edge_dim] replaced by embeddings.
        """
        prot_batch = prot_batch_in.clone()
        lig_batch  = lig_batch_in.clone()

        prot_batch.x         = self.norm_prot_nodes(self.protein_node_enc(prot_batch.x))
        prot_batch.edge_attr = self.norm_prot_edges(self.protein_edge_enc(prot_batch.edge_attr))
        lig_batch.x          = self.norm_lig_nodes(self.ligand_node_enc(lig_batch.x))
        lig_batch.edge_attr  = self.norm_lig_edges(self.ligand_edge_enc(lig_batch.edge_attr))

        return prot_batch, lig_batch
