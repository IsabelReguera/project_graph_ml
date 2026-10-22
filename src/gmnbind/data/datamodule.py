"""DataModule for ComplexDataset, it manages train/val split and DataLoader."""

import pytorch_lightning as pl
from gmnbind.data.complex_dataset import ComplexDataset, complex_collate_fn, build_complex_dataset
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