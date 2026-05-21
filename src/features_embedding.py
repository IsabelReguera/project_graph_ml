import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from utils import DATASET_SCHEMAS

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


class DualGraphTokenizer(nn.Module):
    """
    Tokenizes nodes and edges of both protein and ligand graphs into the same
    latent space using one FeatureTokenizer per tensor.
    Receives and returns PyG Data/Batch objects with updated .x and .edge_attr.
    """

    def __init__(self, token_dim,
                 node_protein_hidden_sizes, edge_protein_hidden_sizes,
                 node_ligand_hidden_sizes, edge_ligand_hidden_sizes):
        """
        Args:
          token_dim: individual projection dimension for each feature.
          node_protein_hidden_sizes: list of ints for the protein node MLP.
            The last element is the final protein node embedding dimension.
          edge_protein_hidden_sizes: list of ints for the protein edge MLP.
          node_ligand_hidden_sizes: list of ints for the ligand node MLP.
          edge_ligand_hidden_sizes: list of ints for the ligand edge MLP.
        """
        super().__init__()

        self.protein_node_enc = FeatureTokenizer(PROTEIN_NODE_SCHEMA, token_dim, node_protein_hidden_sizes)
        self.protein_edge_enc = FeatureTokenizer(PROTEIN_EDGE_SCHEMA, token_dim, edge_protein_hidden_sizes)
        self.ligand_node_enc  = FeatureTokenizer(LIGAND_NODE_SCHEMA,  token_dim, node_ligand_hidden_sizes)
        self.ligand_edge_enc  = FeatureTokenizer(LIGAND_EDGE_SCHEMA,  token_dim, edge_ligand_hidden_sizes)

    def forward(self, prot_batch_in, lig_batch_in):
        """
        Args:
          prot_batch_in, lig_batch_in: PyG Data or Batch with .x and .edge_attr.
        Returns:
          prot_batch, lig_batch: copies with .x [N, node_dim] and .edge_attr [E, edge_dim] replaced by embeddings.
        """
        prot_batch = prot_batch_in.clone()
        lig_batch  = lig_batch_in.clone()

        prot_batch.x         = self.protein_node_enc(prot_batch.x)
        prot_batch.edge_attr = self.protein_edge_enc(prot_batch.edge_attr)
        lig_batch.x          = self.ligand_node_enc(lig_batch.x)
        lig_batch.edge_attr  = self.ligand_edge_enc(lig_batch.edge_attr)

        return prot_batch, lig_batch


class GraphPropLayer(MessagePassing):
    
    """Implementation of a custom message passing layer."""
    
    def __init__(self, node_hidden_sizes, edge_hidden_sizes, aggr_type, update_type):
        
        """Constructor.

        Args:
          node_hidden_sizes: int, dimensionality of the node states.
          edge_hidden_sizes: int, dimensionality of the edge states.
          aggr_type: str, aggregation method for messages, one of {'sum', 'mean', 'max'}.
          update_type: str, type of node update function, one of {'residual', 'gru', 'mlp'}.
        
        Raises:
          ValueError: if aggr_type or update_type are not supported.
        """
        
        if aggr_type not in ['sum', 'mean', 'max']:
            raise ValueError(f"Unknown aggregation type: {aggr_type}")
            
        super(GraphPropLayer, self).__init__(aggr=aggr_type)
        
        self.node_hidden_sizes = node_hidden_sizes
        self.edge_hidden_sizes = edge_hidden_sizes
        self.aggr_type = aggr_type
        self.update_type = update_type

        
        self.message_net = nn.Sequential(
            nn.Linear(node_hidden_sizes * 2 + edge_hidden_sizes, node_hidden_sizes),
            nn.ReLU(),
            nn.Linear(node_hidden_sizes, node_hidden_sizes)
        )
        
      
        if self.update_type in ['residual', 'mlp']:
            self.update_net = nn.Sequential(
                nn.Linear(node_hidden_sizes * 2, node_hidden_sizes),
                nn.ReLU()
            )
        elif self.update_type == 'gru':
            self.update_net = nn.GRUCell(node_hidden_sizes, node_hidden_sizes)
        else:
             raise ValueError(f"Unknown update type: {update_type}")

    def forward(self, x, edge_index, edge_attr):
        """Run one propagation step.

        Args:
          x: [n_nodes, node_hidden_sizes] float tensor, current node states.
          edge_index: [2, n_edges] long tensor, graph connectivity.
          edge_attr: [n_edges, edge_hidden_sizes] float tensor, edge features.

        Returns:
          new_x: [n_nodes, node_hidden_sizes] float tensor, updated node states.
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    
    def message(self, x_i, x_j, edge_attr):
        """Constructs messages from node j to node i.
        
        Automatically called by self.propagate().
        """
        message_inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_net(message_inputs)

    
    def update(self, aggr_out, x): 
        """Updates node states given the aggregated messages.
        
        Automatically called by self.propagate() after message aggregation.
        """
        if self.update_type == 'gru':
            return self.update_net(aggr_out, x)
            
        elif self.update_type == 'residual':
            nuovo_stato = torch.cat([x, aggr_out], dim=-1)
            return x + self.update_net(nuovo_stato)
            
        elif self.update_type == 'mlp':
            nuovo_stato = torch.cat([x, aggr_out], dim=-1)
            return self.update_net(nuovo_stato)