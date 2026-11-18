"""Graph Matching Network: protein and ligand graphs exchange information at every layer."""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import global_mean_pool
from gmnbind.models.layers import GMNLayer
from gmnbind.models.base import BaseAffinityModel

class GraphMatchingNetwork(BaseAffinityModel):
    """
    Full GMN model: protein and ligand graphs communicate through cross-graph attention
    at every layer before being pooled.
    Architecture:
        DualGraphTokenizer -> [GMNLayer x num_layers] -> mean pool -> MLP head -> pAffinity
    """
    def __init__(self, token_dim=32, node_embedding_dims=[128], edge_embedding_dims=[64],
                     num_layers=2, dropout_p=0.1, normalization=True,
                     lr=1e-3, max_lr=1e-3, min_lr=1e-6, l2_norm=0.0,
                     warmup_step_percentage=0.1, use_dynamic_schedule=False):
        
        super().__init__(token_dim, node_embedding_dims, edge_embedding_dims,
                 num_layers, dropout_p, normalization,
                 lr, max_lr, min_lr, l2_norm,
                 warmup_step_percentage, use_dynamic_schedule)

        #GMN layers
        self.gmn_layers = nn.ModuleList([
            GMNLayer(self.node_embedding_dim, self.edge_embedding_dim,
                     dropout_p=dropout_p, use_norm=normalization)
            for _ in range(num_layers)
        ])

        # MLP head for final prediction
        self._build_head()
        
    def forward(self, prot_batch, lig_batch, return_attention=False):
        #Projection on common latent space
        prot, lig = self.tokenizer(prot_batch, lig_batch)

        # propagation GMN (Loop on layers)
        # if return_attention=True, we keep matrix of last layer attention maps for visualization
        attention_maps = None
        for layer in self.gmn_layers:
            layer_out = layer(
                prot.x, prot.edge_index, prot.edge_attr, prot.batch,
                lig.x, lig.edge_index, lig.edge_attr, lig.batch,
                return_attention=return_attention,
            )
            if return_attention:
                prot.x, lig.x, attn_p_to_l, attn_l_to_p = layer_out
                attention_maps = (attn_p_to_l, attn_l_to_p)
            else:
                prot.x, lig.x = layer_out

        #Readout (final mean)
        h_p = global_mean_pool(prot.x, prot.batch)
        h_l = global_mean_pool(lig.x, lig.batch)

        #Final prediction
        out = self.mlp(torch.cat([h_p, h_l], dim=-1)).squeeze(-1)
        if return_attention:
            return out, attention_maps
        return out