import torch
import torch.nn as nn


class DenseFeatureEmbedding(nn.Module):
    """
    alternative to FeatureTokenizer.

    Instead of projecting each feature individually to token_dim and then
    aggregating them with an MLP, this module:
      1) Embeds every categorical feature with a small nn.Embedding
         (dimension `cat_emb_dim`).
      2) Keeps every continuous feature as a scalar.
      3) Concatenates the resulting vector and passes it through a single
         feed-forward MLP that projects it to the final hidden dimension.

    Output shape matches FeatureTokenizer (so DualGraphTokenizer can swap
    them transparently).
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