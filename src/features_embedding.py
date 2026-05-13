import torch
import torch.nn as nn

class GraphEncoder(nn.Module):
    """Encoder module that projects node and edge features to embeddings using feature-specific tokenization."""
    
    def __init__(self, node_schema, edge_schema, token_dim, node_hidden_sizes=None, edge_hidden_sizes=None, name=None):
        """Constructor.

        Args:
          node_schema: list of dicts defining the type and parameters of each node feature column.
          edge_schema: list of dicts defining the type and parameters of each edge feature column.
          token_dim: int, the intermediate projection space for each individual feature (default: 32).
          node_hidden_sizes: list of ints, hidden sizes of node encoder network.
          edge_hidden_sizes: list of ints, hidden sizes of edge encoder network.
          name: name of this module.
        """
        super(GraphEncoder, self).__init__()
        self._node_schema = node_schema
        self._edge_schema = edge_schema
        self._token_dim = token_dim
        self._node_hidden_sizes = node_hidden_sizes if node_hidden_sizes else None
        self._edge_hidden_sizes = edge_hidden_sizes
        
        self._build_model()

    def _build_model(self):
        node_projectors = []
        for feat in self._node_schema:
            if feat['type'] == 'continuous':
                node_projectors.append(nn.Linear(1, self._token_dim))
            elif feat['type'] == 'categorical':
                node_projectors.append(nn.Embedding(feat['num_categories'], self._token_dim))
            else:
                raise ValueError(f"Unknown feature type: {feat['type']}")
        self._node_projectors = nn.ModuleList(node_projectors)

        if self._node_hidden_sizes is not None:
            mlp_input_dim = len(self._node_schema) * self._token_dim
            layer = [nn.Linear(mlp_input_dim, self._node_hidden_sizes[0])]
            for i in range(1, len(self._node_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
            self.MLP1 = nn.Sequential(*layer)
        else:
            self.MLP1 = None


        if self._edge_schema is not None and self._edge_hidden_sizes is not None:
            edge_projectors = []
            for feat in self._edge_schema:
                if feat['type'] == 'continuous':
                    edge_projectors.append(nn.Linear(1, self._token_dim))
                elif feat['type'] == 'categorical':
                    edge_projectors.append(nn.Embedding(feat['num_categories'], self._token_dim))
                else:
                    raise ValueError(f"Unknown feature type: {feat['type']}")
            self._edge_projectors = nn.ModuleList(edge_projectors)

            mlp_input_dim = len(self._edge_schema) * self._token_dim
            layer = [nn.Linear(mlp_input_dim, self._edge_hidden_sizes[0])]
            for i in range(1, len(self._edge_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
            self.MLP2 = nn.Sequential(*layer)
        else:
            self._edge_projectors = None
            self.MLP2 = None

    def _tokenize_features(self, features, schema, projectors):
        """Helper method to slice columns and apply their respective embedding/linear layers."""
        tokens = []
        for i, (feat, proj) in enumerate(zip(schema, projectors)):
            if feat['type'] == 'categorical':
                tokens.append(proj(features[:, i].long()))
            else:
                tokens.append(proj(features[:, i:i + 1]))
        
        return torch.cat(tokens, dim=-1)

    def forward(self, node_features, edge_features=None):
        """Encode node and edge features."""
        
        node_tokens = self._tokenize_features(node_features, self._node_schema, self._node_projectors)
        node_outputs = self.MLP1(node_tokens) if self.MLP1 is not None else node_tokens
            
        
        if edge_features is not None and self._edge_projectors is not None:
            edge_tokens = self._tokenize_features(edge_features, self._edge_schema, self._edge_projectors)
            edge_outputs = self.MLP2(edge_tokens) if self.MLP2 is not None else edge_tokens
        else:
            edge_outputs = edge_features

        return node_outputs, edge_outputs


class DualGraphTokenizer(nn.Module):
    
    """Constructor.
        
        
        Args:
          node_protein_dim: int, dimensionality of the input protein node features
          edge_protein_dim: int, dimensionality of the input protein edge features
          node_ligand_dim: int, dimensionality of the input ligand node features
          edge_ligand_dim: int, dimensionality of the input ligand edge features
          node_protein_hidden_sizes: list of ints, hidden sizes for the protein node encoder network
            The last element is the size of the final node embeddings.
          edge_protein_hidden_sizes: list of ints, hidden sizes for the protein edge encoder network.
            The last element is the size of the final edge embeddings.
          node_ligand_hidden_sizes: list of ints, hidden sizes for the ligand node encoder network.
          edge_ligand_hidden_sizes: list of ints, hidden sizes for the ligand edge encoder network.

    """
    def __init__(self, node_protein_schema, edge_protein_schema, node_ligand_schema, edge_ligand_schema,
                 node_protein_hidden_sizes, edge_protein_hidden_sizes, node_ligand_hidden_sizes, edge_ligand_hidden_sizes,
                 token_dim=32):
        super(DualGraphTokenizer, self).__init__()
        
        self._node_protein_schema = node_protein_schema
        self._edge_protein_schema = edge_protein_schema
        self._node_protein_hidden_sizes = node_protein_hidden_sizes
        self._edge_protein_hidden_sizes = edge_protein_hidden_sizes
        
        self._node_ligand_schema = node_ligand_schema
        self._edge_ligand_schema = edge_ligand_schema
        self._node_ligand_hidden_sizes = node_ligand_hidden_sizes
        self._edge_ligand_hidden_sizes = edge_ligand_hidden_sizes
        self._token_dim = token_dim

        self.protein_encoder = GraphEncoder(
            node_schema=self._node_protein_schema, 
            edge_schema=self._edge_protein_schema,  
            token_dim=self._token_dim,
            node_hidden_sizes=self._node_protein_hidden_sizes,
            edge_hidden_sizes=self._edge_protein_hidden_sizes,
            name='protein-encoder'
        )
        
        self.ligand_encoder = GraphEncoder(
            node_schema=self._node_ligand_schema, 
            edge_schema=self._edge_ligand_schema,  
            token_dim=self._token_dim,
            node_hidden_sizes=self._node_ligand_hidden_sizes,
            edge_hidden_sizes=self._edge_ligand_hidden_sizes,
            name='ligand-encoder'
        )

    def forward(self, prot_batch_in, lig_batch_in):
        """Encode node and edge features for both protein and ligand graphs."""
        prot_batch = prot_batch_in.clone()
        lig_batch = lig_batch_in.clone()
        
        prot_x, prot_edge = self.protein_encoder(prot_batch.x, prot_batch.edge_attr)
        prot_batch.x = prot_x
        prot_batch.edge_attr = prot_edge
        
        lig_x, lig_edge = self.ligand_encoder(lig_batch.x, lig_batch.edge_attr)
        lig_batch.x = lig_x
        lig_batch.edge_attr = lig_edge
        
        return prot_batch, lig_batch





import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

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