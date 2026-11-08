"""Baseline: two independent GNNs (protein and ligand) with no communication between graphs."""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import GINEConv, global_mean_pool
from gmnbind.models.base import BaseAffinityModel

class BaselineGNN(BaseAffinityModel):
    """
    Baseline: two independent GNNs (protein and ligand), no communication between the graphs.
    Same tokenizer, pooling and MLP head as the GMN: only the propagation differs.
    Architecture:
        DualGraphTokenizer -> [GINEConv x num_layers] (per graph) -> mean pool -> MLP head -> pAffinity
    """
    def __init__(self, token_dim=32, node_embedding_dims=[128], edge_embedding_dims=[64],
                     num_layers=2, dropout_p=0.1, normalization=True,
                     lr=1e-3, max_lr=1e-3, min_lr=1e-6, l2_norm=0.0,
                     warmup_step_percentage=0.1, use_dynamic_schedule=False):
        
        super().__init__(token_dim, node_embedding_dims, edge_embedding_dims,
                 num_layers, dropout_p, normalization,
                 lr, max_lr, min_lr, l2_norm,
                 warmup_step_percentage, use_dynamic_schedule)
        
        #GNN layers for protein and ligand
        self.prot_gnn_layers = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
                    nn.ReLU(),
                    nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
                ), edge_dim=self.edge_embedding_dim
            ) for _ in range(num_layers) 
        ])
        
        self.lig_gnn_layers  = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
                    nn.ReLU(),
                    nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
                ), edge_dim=self.edge_embedding_dim
            ) for _ in range(num_layers)
        ])
        
        #MLP head for final prediction
        self._build_head()
        
    def forward(self, prot_batch, lig_batch):
        prot, lig = self.tokenizer(prot_batch, lig_batch)

        for layer in self.prot_gnn_layers:
            prot.x = layer(prot.x, prot.edge_index, prot.edge_attr)
        for layer in self.lig_gnn_layers:
            lig.x = layer(lig.x, lig.edge_index, lig.edge_attr)
       
        # mean pooling: [N_nodes, emb_dim] -> [batch_size, emb_dim]
        h_p = global_mean_pool(prot.x, prot.batch)
        h_l = global_mean_pool(lig.x, lig.batch)

        return self.mlp(torch.cat([h_p, h_l], dim=-1)).squeeze(-1)
 