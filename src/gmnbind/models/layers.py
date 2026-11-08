"""Building blocks of the Graph Matching Network: cross-graph attention and the GMN layer."""
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

class CrossGraphAttention(nn.Module):
    """
    Bidirectional cross-attention between protein nodes and ligand nodes
    (scaled dot-product attention). Handles PyG batches by converting them to dense tensors.
    """
    def __init__(self, node_dim):
        super().__init__()
        # linear projections for query, key and value (shared between protein and ligand)
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.scale = node_dim ** -0.5  # 1/sqrt(d) scaling of the scores

    def forward(self, x_prot, batch_prot, x_lig, batch_lig, return_attention=False):
        """
        x_prot: [N_tot_prot, dim]  nodes of all the proteins in the batch
        x_lig:  [N_tot_lig, dim]   nodes of all the ligands in the batch
        return_attention: if True, also return the attention matrices

        Returns the cross-graph context of every node (att_prot, att_lig), i.e. what is
        subtracted from the node state in the GMN layer.
        If return_attention=True: (att_prot, att_lig, attn_p_to_l, attn_l_to_p)
            attn_p_to_l: [B, N_prot, N_lig]  attention of each protein node over the ligand nodes
            attn_l_to_p: [B, N_lig, N_prot]  attention of each ligand node over the protein nodes
        """
        # 1. PyG batches -> dense tensors [B, max_nodes, dim]
        #    mask marks the real nodes vs the padding added to equalize lengths
        p_dense, p_mask = to_dense_batch(x_prot, batch_prot)
        l_dense, l_mask = to_dense_batch(x_lig, batch_lig)

        # 2. Q, K, V projections
        q_p = self.q_proj(p_dense)  # protein query
        k_p = self.k_proj(p_dense)  # protein key
        v_p = self.v_proj(p_dense)  # protein value

        q_l = self.q_proj(l_dense)  # ligand query
        k_l = self.k_proj(l_dense)  # ligand key
        v_l = self.v_proj(l_dense)  # ligand value

        # 3. Similarity scores (scaled dot product), protein -> ligand (Li et al. 2019, Eq. 10)
        #    torch.bmm = batched matrix multiplication
        scores_p_to_l = torch.bmm(q_p, k_l.transpose(1, 2)) * self.scale

        # padding: -inf scores towards padded ligand nodes
        scores_p_to_l.masked_fill_(~l_mask.unsqueeze(1), float('-inf'))

        # softmax -> attention weights
        attn_p_to_l = torch.softmax(scores_p_to_l, dim=-1)

        # 4. Cross-graph context: weighted sum of ligand values (Li et al. 2019, Eq. 11)
        message_from_l = torch.bmm(attn_p_to_l, v_l)

        # --- Same in the other direction: ligand -> protein ---
        scores_l_to_p = torch.bmm(q_l, k_p.transpose(1, 2)) * self.scale
        scores_l_to_p.masked_fill_(~p_mask.unsqueeze(1), float('-inf'))
        attn_l_to_p = torch.softmax(scores_l_to_p, dim=-1)
        message_from_p = torch.bmm(attn_l_to_p, v_p)

        # 5. Dense tensors -> flat PyG format (mask.view(-1) selects the real nodes)
        att_prot = message_from_l.view(-1, message_from_l.size(-1))[p_mask.view(-1)]
        att_lig = message_from_p.view(-1, message_from_p.size(-1))[l_mask.view(-1)]

        if return_attention:
            return att_prot, att_lig, attn_p_to_l, attn_l_to_p
        return att_prot, att_lig


class GMNLayer(nn.Module):
    """
    One Graph Matching Network layer (Li et al. 2019, Eq. 5):
    intra-graph message passing (GINEConv) + cross-graph attention,
    node update from [h_i, intra-graph message, h_i - cross-graph context].
    """
    def __init__(self, node_dim, edge_dim, dropout_p=0.1, use_norm=True):
        super().__init__()

        # 1. Intra-graph message passing (same as the baseline)
        self.intra_prot = GINEConv(
            nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, node_dim)
            ), edge_dim=edge_dim
        )
        self.intra_lig = GINEConv(
            nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, node_dim)
            ), edge_dim=edge_dim
        )

        # 2. Cross-graph matching
        self.cross_attention = CrossGraphAttention(node_dim)

        # 3. Node update from [current state, neighbor messages, current state - cross-graph context]
        #    Linear -> [LayerNorm] -> ReLU
        def _update_mlp():
            layers = [nn.Linear(node_dim * 3, node_dim)]
            if use_norm:
                layers.append(nn.LayerNorm(node_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout_p))  # removed: dropout as last op here made training unstable
            return nn.Sequential(*layers)

        self.update_prot = _update_mlp()
        self.update_lig  = _update_mlp()

    def forward(self, x_prot, edge_idx_prot, edge_attr_prot, batch_prot,
                      x_lig, edge_idx_lig, edge_attr_lig, batch_lig, return_attention=False):

        # A: intra-graph messages
        msg_prot = self.intra_prot(x_prot, edge_idx_prot, edge_attr_prot)
        msg_lig = self.intra_lig(x_lig, edge_idx_lig, edge_attr_lig)

        # B: cross-graph attention
        cross_out = self.cross_attention(x_prot, batch_prot, x_lig, batch_lig,
                                         return_attention=return_attention)
        if return_attention:
            att_prot, att_lig, attn_p_to_l, attn_l_to_p = cross_out
        else:
            att_prot, att_lig = cross_out

        # C: matching vector h_i - sum_j(a_ij * v_j)
        diff_prot = x_prot - att_prot
        diff_lig = x_lig - att_lig

        # D: node update from [previous state, neighbor messages, cross-graph difference]
        new_x_prot = self.update_prot(torch.cat([x_prot, msg_prot, diff_prot], dim=-1))
        new_x_lig = self.update_lig(torch.cat([x_lig, msg_lig, diff_lig], dim=-1))

        if return_attention:
            return new_x_prot, new_x_lig, attn_p_to_l, attn_l_to_p
        return new_x_prot, new_x_lig