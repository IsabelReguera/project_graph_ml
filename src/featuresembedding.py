import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GraphEncoder(nn.Module):
    
    """Encoder module that projects node and edge features to some embeddings."""
    
    def __init__(self, node_feature_dim, edge_feature_dim, node_hidden_sizes=None, edge_hidden_sizes=None, name=None):
        
        """Constructor.

        Args:
          node_hidden_sizes: if provided should be a list of ints, hidden sizes of
            node encoder network, the last element is the size of the node outputs.
            If not provided, node features will pass through as is.
          edge_hidden_sizes: if provided should be a list of ints, hidden sizes of
            edge encoder network, the last element is the size of the edge outptus.
            If not provided, edge features will pass through as is.
          name: name of this module.
        """

        super(GraphEncoder, self).__init__()
        self._node_feature_dim = node_feature_dim
        self._edge_feature_dim = edge_feature_dim
        self._node_hidden_sizes = node_hidden_sizes if node_hidden_sizes else None
        self._edge_hidden_sizes = edge_hidden_sizes
        self._build_model()

    def _build_model(self):
        layer = []
        layer.append(nn.Linear(self._node_feature_dim, self._node_hidden_sizes[0]))
        for i in range(1, len(self._node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
        self.MLP1 = nn.Sequential(*layer)

        if self._edge_hidden_sizes is not None:
            layer = []
            layer.append(nn.Linear(self._edge_feature_dim, self._edge_hidden_sizes[0]))
            for i in range(1, len(self._edge_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
            self.MLP2 = nn.Sequential(*layer)
        else:
            self.MLP2 = None

    def forward(self, node_features, edge_features=None):
        
        """Encode node and edge features.

        Args:
          node_features: [n_nodes, node_feat_dim] float tensor.
          edge_features: if provided, should be [n_edges, edge_feat_dim] float
            tensor.

        Returns:
          node_outputs: [n_nodes, node_embedding_dim] float tensor, node embeddings.
          edge_outputs: if edge_features is not None and edge_hidden_sizes is not
            None, this is [n_edges, edge_embedding_dim] float tensor, edge
            embeddings; otherwise just the input edge_features.
        """
        if self._node_hidden_sizes is None:
            node_outputs = node_features
        else:
            node_outputs = self.MLP1(node_features)
            
        if edge_features is None or self._edge_hidden_sizes is None:
            edge_outputs = edge_features
        else:
            edge_outputs = self.MLP2(edge_features)

        return node_outputs, edge_outputs



class DualGraphTokenizer(nn.Module):
    def __init__(self,node_protein_dim,edge_protein_dim,node_ligand_dim,edge_ligand_dim,node_protein_hidden_sizes, edge_protein_hidden_sizes,node_ligand_hidden_sizes, edge_ligand_hidden_sizes):
        
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

        super(DualGraphTokenizer, self).__init__()
        
        self._node_protein_dim = node_protein_dim
        self._edge_protein_dim = edge_protein_dim
        self._node_protein_hidden_sizes = node_protein_hidden_sizes
        self._edge_protein_hidden_sizes = edge_protein_hidden_sizes
        
        self._node_ligand_dim = node_ligand_dim
        self._edge_ligand_dim = edge_ligand_dim
        self._node_ligand_hidden_sizes = node_ligand_hidden_sizes
        self._edge_ligand_hidden_sizes = edge_ligand_hidden_sizes
        

        self.protein_encoder = GraphEncoder(
            node_feature_dim=self._node_protein_dim, 
            edge_feature_dim=self._edge_protein_dim,  
            node_hidden_sizes=self._node_protein_hidden_sizes,
            edge_hidden_sizes=self._edge_protein_hidden_sizes,
            name='protein-encoder'
        )
        
        
        self.ligand_encoder = GraphEncoder(
            node_feature_dim=self._node_ligand_dim, 
            edge_feature_dim=self._edge_ligand_dim,  
            node_hidden_sizes=self._node_ligand_hidden_sizes,
            edge_hidden_sizes=self._edge_ligand_hidden_sizes,
            name='ligand-encoder'
        )

    def forward(self, prot_batch_in, lig_batch_in):
        
        """Encode node and edge features for both protein and ligand graphs.

        Args:
          prot_batch: PyTorch Geometric Data or Batch object representing the protein. 
            Expected to contain `.x` [n_prot_nodes, node_protein_sizes] float tensor and 
            `.edge_attr` [n_prot_edges, edge_protein_sizes] float tensor.
          lig_batch: PyTorch Geometric Data or Batch object representing the ligand. 
            Expected to contain `.x` [n_lig_nodes, node_ligand_sizes] float tensor and 
            `.edge_attr` [n_lig_edges, edge_ligand_sizes] float tensor.

        Returns:
          prot_batch: PyTorch Geometric Data or Batch object, where `.x` and `.edge_attr` 
            have been updated to their new embedding dimensions.
          lig_batch: PyTorch Geometric Data or Batch object, where `.x` and `.edge_attr` 
            have been updated to their new embedding dimensions.
        """

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