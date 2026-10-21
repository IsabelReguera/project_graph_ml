import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

class CrossGraphAttention(nn.Module):
    """
    Calcola la cross-attention tra i nodi della proteina e i nodi del ligando.
    Gestisce automaticamente lo 'spacchettamento' dei batch di PyG.
    """
    def __init__(self, node_dim):
        super().__init__()
        # Usiamo pesi lineari per proiettare Query, Key e Value (come nei Transformer standard)
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.scale = node_dim ** -0.5 # Fattore di scaling per la softmax

    def forward(self, x_prot, batch_prot, x_lig, batch_lig, return_attention=False):
        """
        x_prot: [N_tot_prot, dim] - Nodi di tutte le proteine nel batch
        x_lig:  [N_tot_lig, dim]  - Nodi di tutti i ligandi nel batch
        return_attention: se True restituisce anche le matrici di attenzione grezze
        Restituisce i vettori di attenzione da sottrarre: att_prot e att_lig
        Se return_attention=True: (att_prot, att_lig, attn_p_to_l, attn_l_to_p)
            attn_p_to_l: [B, N_prot, N_lig] - quanto ogni nodo proteina guarda ogni nodo ligando
            attn_l_to_p: [B, N_lig, N_prot] - quanto ogni nodo ligando guarda ogni nodo proteina
        """
        # 1. 'Srotoliamo' i batch PyG in tensori densi [Batch, Max_Nodes, Dim]
        # mask ci dice quali nodi sono veri e quali sono padding (aggiunti per pareggiare le lunghezze)
        p_dense, p_mask = to_dense_batch(x_prot, batch_prot)
        l_dense, l_mask = to_dense_batch(x_lig, batch_lig)

        # 2. Proiezioni Q, K, V
        q_p = self.q_proj(p_dense) # Query Proteina
        k_p = self.k_proj(p_dense) # Key Proteina
        v_p = self.v_proj(p_dense) # Value Proteina

        q_l = self.q_proj(l_dense) # Query Ligando
        k_l = self.k_proj(l_dense) # Key Ligando
        v_l = self.v_proj(l_dense) # Value Ligando

        # 3. Calcolo Punteggi di Affinità (Dot Product Similarity)
        # Proteina guarda Ligando (Equazione 10 del paper)
        # torch.bmm fa la moltiplicazione tra matrici tenendo conto della dimensione Batch
        scores_p_to_l = torch.bmm(q_p, k_l.transpose(1, 2)) * self.scale

        # Gestione del padding: mettiamo a -infinito i punteggi verso nodi finti
        scores_p_to_l.masked_fill_(~l_mask.unsqueeze(1), float('-inf'))

        # Softmax per avere le percentuali (pesi di attenzione)
        attn_p_to_l = torch.softmax(scores_p_to_l, dim=-1)

        # 4. Creazione dell'Identikit (Equazione 11 del paper)
        # La proteina "tira a sé" le features del ligando
        message_from_l = torch.bmm(attn_p_to_l, v_l)

        # --- Facciamo la stessa cosa al contrario (Ligando guarda Proteina) ---
        scores_l_to_p = torch.bmm(q_l, k_p.transpose(1, 2)) * self.scale
        scores_l_to_p.masked_fill_(~p_mask.unsqueeze(1), float('-inf'))
        attn_l_to_p = torch.softmax(scores_l_to_p, dim=-1)
        message_from_p = torch.bmm(attn_l_to_p, v_p)

        # 5. Rimpacchettiamo i tensori densi nel formato piatto di PyG
        # mask.view(-1) ritrova esattamente gli indici dei nodi veri
        att_prot = message_from_l.view(-1, message_from_l.size(-1))[p_mask.view(-1)]
        att_lig = message_from_p.view(-1, message_from_p.size(-1))[l_mask.view(-1)]

        if return_attention:
            return att_prot, att_lig, attn_p_to_l, attn_l_to_p
        return att_prot, att_lig


class GMNLayer(nn.Module):
    """
    Un singolo strato della Graph Matching Network (Equazione 5 del paper).
    Esegue il message passing interno (GINEConv) e poi lo incrocia con la Cross-Attention.
    """
    def __init__(self, node_dim, edge_dim, dropout_p=0.1, use_norm=True):
        super().__init__()

        # 1. Message Passing Interno (uguale alla baseline)
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

        # 2. Matching Esterno
        self.cross_attention = CrossGraphAttention(node_dim)

        # 3. Aggiornamento finale del nodo
        # Prende in input: Nodo attuale + Messaggi Vicini + (Nodo attuale - Attenzione Opposta)
        # Linear -> [LayerNorm]-> ReLU-> [Dropout]
        def _update_mlp():
            layers = [nn.Linear(node_dim * 3, node_dim)]
            if use_norm:
                layers.append(nn.LayerNorm(node_dim))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(dropout_p)) dropout come ultimo layer qua può fare casino
            return nn.Sequential(*layers)

        self.update_prot = _update_mlp()
        self.update_lig  = _update_mlp()

    def forward(self, x_prot, edge_idx_prot, edge_attr_prot, batch_prot,
                      x_lig, edge_idx_lig, edge_attr_lig, batch_lig, return_attention=False):

        # Step A: Parla coi vicini (Intra-graph)
        msg_prot = self.intra_prot(x_prot, edge_idx_prot, edge_attr_prot)
        msg_lig = self.intra_lig(x_lig, edge_idx_lig, edge_attr_lig)

        # Step B: Guarda l'altra molecola (Inter-graph)
        cross_out = self.cross_attention(x_prot, batch_prot, x_lig, batch_lig,
                                         return_attention=return_attention)
        if return_attention:
            att_prot, att_lig, attn_p_to_l, attn_l_to_p = cross_out
        else:
            att_prot, att_lig = cross_out

        # Step C: Calcola la differenza (La genialata del paper: h_i - sum(a * h_j))
        diff_prot = x_prot - att_prot
        diff_lig = x_lig - att_lig

        # Step D: Aggiornamento finale fondendo tutto
        # Concateniamo: [Stato Precedente, Messaggi dai Vicini, Differenza con l'altro grafo]
        new_x_prot = self.update_prot(torch.cat([x_prot, msg_prot, diff_prot], dim=-1))
        new_x_lig = self.update_lig(torch.cat([x_lig, msg_lig, diff_lig], dim=-1))

        if return_attention:
            return new_x_prot, new_x_lig, attn_p_to_l, attn_l_to_p
        return new_x_prot, new_x_lig