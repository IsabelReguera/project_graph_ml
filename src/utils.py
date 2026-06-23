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
    {'type': 'categorical' , 'num_categories': 5},  # tipo legame (0/1/2/3/4) (1=singolo, 2=doppio, 3=triplo, 4=aromatico, 0=falso/distanza)
    {'type': 'categorical', 'num_categories':2 },   # coniugazione (0=falso, 1=vero)
    {'type': 'continuous'},                         # distanza 3D in Å
]

DATASET_SCHEMAS = {
    'protein_node': PROTEIN_NODE_SCHEMA,
    'protein_edge': PROTEIN_EDGE_SCHEMA,
    'ligand_node': LIGAND_NODE_SCHEMA,
    'ligand_edge': LIGAND_EDGE_SCHEMA
}

# ---------------------------------------------------------------------------
# Lightning Data Module
# ---------------------------------------------------------------------------

import pytorch_lightning as pl
from complex_dataset import ComplexDataset, complex_collate_fn, build_complex_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class ComplexDataModule(pl.LightningDataModule):
    """LightningDataModule for ComplexDataset, it manages train/val split and DataLoader."""
    
    def __init__(self, processed_dir, csv_path, batch_size=16, limit=None, train_val_split=0.2, seed=47):
        super().__init__()
        self.processed_dir = processed_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.limit = limit
        self.train_val_split = train_val_split
        self.seed = seed
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_dataset = build_complex_dataset(self.processed_dir, self.csv_path, limit=self.limit)
            all_codes = full_dataset.pdb_codes

            # Split train/val with PDB code
            train_codes, val_codes = train_test_split(all_codes, test_size=self.train_val_split, random_state=self.seed)

            self.train_ds = ComplexDataset(self.processed_dir, train_codes)
            self.val_ds = ComplexDataset(self.processed_dir, val_codes)
    
            print(f"Train: {len(self.train_ds)} complexes | Val: {len(self.val_ds)} complexes")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=complex_collate_fn, 
            num_workers=4,
            persistent_workers=False 
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=complex_collate_fn, 
            num_workers=4, 
            persistent_workers=False
        )