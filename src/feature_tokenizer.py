import torch
import torch.nn as nn
from utils import DATASET_SCHEMAS


# ---------------------------------------------------------------------------
# Schemi feature -> un dizionario per ogni feature, nell'ordine delle colonne
# ---------------------------------------------------------------------------

PROTEIN_NODE_SCHEMA = DATASET_SCHEMAS['protein_node']
LIGAND_NODE_SCHEMA = DATASET_SCHEMAS['ligand_node']
PROTEIN_EDGE_SCHEMA = DATASET_SCHEMAS['protein_edge']
LIGAND_EDGE_SCHEMA = DATASET_SCHEMAS['ligand_edge']

# ---------------------------------------------------------------------------
# FeatureTokenizer
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """
    Espande ogni feature individualmente in uno spazio token_dim,
    concatena tutti i token e li aggrega con un MLP a due layer.

    Feature continue:   Linear(1, token_dim)            
    Feature categoriche: nn.Embedding(num_categories, token_dim)

    Aggregazione: Linear(N·token_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, hidden_dim) 
    Output: [N_nodes, hidden_dim]
    """

    def __init__(self, schema, token_dim, hidden_dim):
        super().__init__()
        self.schema = schema
        n = len(schema)

        projectors = []
        for feat in schema:
            if feat['type'] == 'continuous':
                projectors.append(nn.Linear(1, token_dim))
            else:
                projectors.append(nn.Embedding(feat['num_categories'], token_dim))
        self.projectors = nn.ModuleList(projectors)

        self.aggregator = nn.Sequential(
            nn.Linear(n * token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        """
        x: [N_nodes, N_features] float tensor.
           Le colonne categoriche sono float ma vengono convertite a long internamente.
        """
        tokens = []
        for i, (feat, proj) in enumerate(zip(self.schema, self.projectors)): # 
            if feat['type'] == 'categorical':
                tokens.append(proj(x[:, i].long()))   #  [N] -> [N, token_dim] 
            else:
                tokens.append(proj(x[:, i:i + 1]))    # [N,1]  -> [N, token_dim]

        return self.aggregator(torch.cat(tokens, dim=-1))  # concateno -> [N, hidden_dim] 


# ---------------------------------------------------------------------------
# DualFeatureTokenizer -> wrapper per proteina + ligando
# ---------------------------------------------------------------------------

class DualFeatureTokenizer(nn.Module):
    """
    Tokenizza nodi e archi di proteina e ligando nello stesso spazio latente
    usando FeatureTokenizer per tutti e quattro i tensori.
    Riceve e restituisce oggetti PyG Data/Batch con .x e .edge_attr aggiornati.
    """

    def __init__(self, token_dim=32, node_embedding_dim=128, edge_embedding_dim=64):
        super().__init__()
        self.protein_node_tok = FeatureTokenizer(PROTEIN_NODE_SCHEMA, token_dim, node_embedding_dim)
        self.ligand_node_tok  = FeatureTokenizer(LIGAND_NODE_SCHEMA,  token_dim, node_embedding_dim)
        self.protein_edge_tok = FeatureTokenizer(PROTEIN_EDGE_SCHEMA, token_dim, edge_embedding_dim)
        self.ligand_edge_tok  = FeatureTokenizer(LIGAND_EDGE_SCHEMA,  token_dim, edge_embedding_dim)

    def forward(self, prot_batch_in, lig_batch_in):
        """
        prot_batch_in, lig_batch_in: PyG Data o Batch con .x e .edge_attr.
        Restituisce copie con .x [N, node_embedding_dim] e .edge_attr [E, edge_embedding_dim].
        """
        prot_batch = prot_batch_in.clone()
        lig_batch  = lig_batch_in.clone()

        prot_batch.x         = self.protein_node_tok(prot_batch.x)
        prot_batch.edge_attr = self.protein_edge_tok(prot_batch.edge_attr)

        lig_batch.x          = self.ligand_node_tok(lig_batch.x)
        lig_batch.edge_attr  = self.ligand_edge_tok(lig_batch.edge_attr)

        return prot_batch, lig_batch


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import os
    import glob

    processed_dir = os.path.join('data', 'processed')
    prot_files = sorted(glob.glob(os.path.join(processed_dir, 'protein_*.pt')))
    if not prot_files:
        print("Nessun file processato trovato in data/processed. Eseguire prima pdbbind_dataset.py.")
        exit(1)

    pdb_code = os.path.basename(prot_files[0]).replace('protein_', '').replace('.pt', '')
    lig_path = os.path.join(processed_dir, f'ligand_{pdb_code}.pt')
    if not os.path.exists(lig_path):
        print(f"Ligando mancante per {pdb_code}.")
        exit(1)

    prot_data = torch.load(os.path.join(processed_dir, f'protein_{pdb_code}.pt'), weights_only=False)
    lig_data  = torch.load(lig_path, weights_only=False)

    print(f"Complesso: {pdb_code}")
    print(f"  Input — proteina nodi: {prot_data.x.shape}, archi: {prot_data.edge_attr.shape}")
    print(f"  Input — ligando  nodi: {lig_data.x.shape},  archi: {lig_data.edge_attr.shape}")

    tok = DualFeatureTokenizer(token_dim=32, node_embedding_dim=128, edge_embedding_dim=64)
    prot_out, lig_out = tok(prot_data, lig_data)

    print(f"  Output — proteina nodi: {prot_out.x.shape}, archi: {prot_out.edge_attr.shape}")
    print(f"  Output — ligando  nodi: {lig_out.x.shape},  archi: {lig_out.edge_attr.shape}")
