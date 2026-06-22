import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

from complex_dataset import ComplexDataset, complex_collate_fn, build_complex_dataset
from baseline import BaselineGNN
from graphmatchingnetwork import GraphMatchingNetwork


class ComplexDataModule(pl.LightningDataModule):
    """LightningDataModule for ComplexDataset, it manages train/val split and DataLoader."""
    
    def __init__(self, processed_dir, csv_path, batch_size=16, limit=None, train_val_split=0.2,seed=47):
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
            persistent_workers=True 
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=complex_collate_fn, 
            num_workers=4, 
            persistent_workers=True
        )

#per migliorare le prestazioni su GPU 
torch.set_float32_matmul_precision('medium')

def main(config_path='config.yaml'):

    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    processed_dir = config['data']['processed_dir']
    csv_path      = config['data']['csv_path']
    batch_size    = config['training']['batch_size']

    data_module = ComplexDataModule(
        processed_dir=processed_dir,
        csv_path=csv_path,
        batch_size=batch_size,
        limit=config['data'].get('limit', None),
        train_val_split=0.2,
        seed=config['data'].get('seed', 47),
    )

    model = BaselineGNN(
        token_dim=config['model']['token_dim'],
        node_embedding_dims=[config['model']['node_embedding_dim']],
        edge_embedding_dims=[config['model']['edge_embedding_dim']],
        dropout_p=0.0, # Per la baseline, niente dropout
        normalization=False, # LayerNorm prima di ReLU
        lr=config['model']['lr'],
    )
    
    """model = GraphMatchingNetwork(
        token_dim=config['model']['token_dim'],
        node_embedding_dim=[config['model']['node_embedding_dim']],
        edge_embedding_dim=[config['model']['edge_embedding_dim']],
        num_layers=2,
        lr=config['model']['lr'],
        dropout_p=config['model']['dropout_p'],
        normalization=config['model']['normalization'],
    )"""
    
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        precision='bf16-mixed',
        log_every_n_steps=10,
    )

    trainer.fit(
        model, 
        datamodule=data_module
    )


if __name__ == '__main__':
    #PER RUNNARE -> cartella repository in terminale, poi: python src/train.py
    main()
