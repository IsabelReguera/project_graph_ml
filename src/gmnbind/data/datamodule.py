"""DataModule for ComplexDataset: train/val/test split and DataLoaders."""

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from gmnbind.data.complex_dataset import ComplexDataset, complex_collate_fn
from gmnbind.data.splits import load_split

class ComplexDataModule(pl.LightningDataModule):
    """
    Reads train/val/(test) PDB codes from split_dir (data/index/splits/...)
    and loads the matching protein/ligand graphs from processed_dir.
    """
    
    def __init__(self, processed_dir, split_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.split_dir = Path(split_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ds = self.val_ds = self.test_ds = None
        
    def _existing(self, codes, name):
        """Keep only codes with both .pt files; warn about the missing ones."""
        ok = [c for c in codes
              if (self.processed_dir / f"protein_{c}.pt").exists()
              and (self.processed_dir / f"ligand_{c}.pt").exists()]
        if len(ok) < len(codes):
            print(f"WARNING: {len(codes) - len(ok)}/{len(codes)} {name} complexes have no .pt and are skipped")
        return ok
    
    def setup(self, stage=None):
        if self.train_ds is not None:          # already set up (fit -> test)
            return
        
        split = load_split(self.split_dir)
        self.train_ds = ComplexDataset(self.processed_dir, self._existing(split["train"], "train"))
        self.val_ds = ComplexDataset(self.processed_dir, self._existing(split["val"], "val"))
        if "test" in split:
            self.test_ds = ComplexDataset(self.processed_dir, self._existing(split["test"], "test"))
        print(f"Split {self.split_dir.name}: train {len(self.train_ds)} | val {len(self.val_ds)} | "
              f"test {len(self.test_ds) if self.test_ds else 0}")
        
    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            collate_fn=complex_collate_fn,
            num_workers=self.num_workers, 
            persistent_workers=False
        )
        
    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        if self.test_ds is None:
            raise RuntimeError(f"Split {self.split_dir.name} has no test set.")
        return self._loader(self.test_ds, shuffle=False)