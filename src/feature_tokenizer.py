import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Schemi feature -> un dizionario per ogni feature, nell'ordine delle colonne
# ---------------------------------------------------------------------------

PROTEIN_NODE_SCHEMA = [
    {'type': 'continuous'},                           # meiler 0 — idrofobicità
    {'type': 'continuous'},                           # meiler 1 — volume sterico
    {'type': 'continuous'},                           # meiler 2 — polarizzabilità
    {'type': 'continuous'},                           # meiler 3 — polarità
    {'type': 'continuous'},                           # meiler 4 — pI
    {'type': 'continuous'},                           # meiler 5 — indice di rifrazione
    {'type': 'continuous'},                           # meiler 6 — propensione α-elica
    {'type': 'continuous'},                           # b_factor
    {'type': 'categorical', 'num_categories': 21}     # tipo aminoacido (20 standard + 1 per residui non standard/virtuali)
]

LIGAND_NODE_SCHEMA = [
    {'type': 'categorical', 'num_categories': 120},   # numero atomico
    {'type': 'continuous'},                           # degree
    {'type': 'continuous'},                           # H totali TODO: valutare se lasciare come continua o trasformare in categorica (0/1/2/3+)
    {'type': 'continuous'},                           # carica formale
    {'type': 'categorical', 'num_categories': 2},     # aromaticità (0=falso, 1=vero)
    {'type': 'categorical', 'num_categories': 2},     # in-ring (0=falso, 1=vero)
    {'type': 'categorical', 'num_categories': 7},     # ibridazione
]

PROTEIN_EDGE_SCHEMA = [
    {'type': 'categorical', 'num_categories': 2},   # tipo legame (0=contatto spaziale, 1=peptide bond)
    {'type': 'continuous'},                         # distanza Cα–Cα in Å
]

LIGAND_EDGE_SCHEMA = [
    {'type': 'categorical' , 'num_categories': 5},  # tipo legame (0/2/3/4/6) (2=singolo, 3=arom, 4=doppio, 6=triplo, 0=falso/distanza)
    {'type': 'categorical', 'num_categories':2 },   # coniugazione (0=falso, 1=vero)
    {'type': 'continuous'},                         # distanza 3D in Å
]


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
        Restituisce copie con .x [N, hidden_dim] e .edge_attr [E, edge_dim].
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
