import os
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
import torch

from complex_dataset import ComplexDataset, complex_collate_fn, build_complex_dataset
from baseline import BaselineGNN
from graphmatchingnetwork import GraphMatchingNetwork

#per migliorare le prestazioni su GPU 
torch.set_float32_matmul_precision('medium')


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


def main(config_path='config.yaml'):

    # --- Load configuration from YAML file --------
    with open(config_path) as f:
        config = yaml.safe_load(f)

    processed_dir   = config['data']['processed_dir']
    log_dir        = config['data']['log_dir']
    csv_path        = config['data']['csv_path']
    batch_size      = config['training']['batch_size']
    model_type      = config['model'].get('model_type', 'baseline')  # Default to 'baseline' if not specified
    experiment_name = config['model']['experiment_name']
    
    token_dim = config['model']['token_dim']
    node_embedding_dims = config['model']['node_embedding_dims']
    edge_embedding_dims = config['model']['edge_embedding_dims']
    num_layers = config['model']['num_layers']
    lr = config['model']['lr']
    dropout_p = config['model']['dropout_p']
    normalization = config['model']['normalization']


    # --- Setup training environment --------
    
    logger = CSVLogger(save_dir=log_dir, name=experiment_name)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=logger.log_dir,
            filename='best',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
        ),
        EarlyStopping(monitor='val_loss', patience=config['training']['patience'], mode='min'),
    ]

    data_module = ComplexDataModule(
        processed_dir=processed_dir,
        csv_path=csv_path,
        batch_size=batch_size,
        limit=config['data'].get('limit', None),
        train_val_split=0.2,
        seed=config['data']['seed'],
    )

    if model_type == 'baseline':
        model = BaselineGNN(
            token_dim=token_dim,
            node_embedding_dims=node_embedding_dims,
            edge_embedding_dims=edge_embedding_dims,
            dropout_p=0.0,
            normalization=False,
            lr=lr,
        )
    else:
        model = GraphMatchingNetwork(
            token_dim=token_dim,
            node_embedding_dims=node_embedding_dims,
            edge_embedding_dims=edge_embedding_dims,
            num_layers=num_layers,
            lr=lr,
            dropout_p=dropout_p,
            normalization=normalization,
        )

    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        precision='bf16-mixed',
        log_every_n_steps=batch_size,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=data_module)
    
    
    
    # --- LOGGING METRICS ------
    metrics = pd.read_csv(os.path.join(logger.log_dir, 'metrics.csv'))
    best_idx = metrics['val_loss'].idxmin()
    best_row = metrics.loc[best_idx]
    best_val = best_row['val_loss']
    best_epoch = best_row['epoch']
    best_val_mae = best_row['val_mae']
    best_val_pearson = best_row['val_pearson_r']
    print(f"[{experiment_name}]  best val_loss = {best_val:.4f} | R = {best_val_pearson:.4f} | epoch = {best_epoch} ")
    
    summary=[]
    summary.append({
        'name':                 experiment_name+f"_{trainer.logger.version}",
        'model':                model_type,
        'best_val_loss':        best_val,
        'best_val_mae':         best_val_mae,
        'pearson_r':            best_val_pearson,
        'stopped_epoch':        best_epoch,
        'node_embedding_dims':  node_embedding_dims[-1],
        'edge_embedding_dims':  edge_embedding_dims[-1],
        'num_layers':           num_layers,
        'normalization':        normalization,
        'dropout_p':            dropout_p,
        'lr':                   lr,
        'batch_size':           batch_size,
    })
    
    # Save summary to CSV 
    summary_path = os.path.join(log_dir, 'summary.csv')
    if os.path.exists(summary_path):
        existing_summary = pd.read_csv(summary_path)
        summary_df = pd.DataFrame(summary)
        summary_df = pd.concat([existing_summary, summary_df], ignore_index=True)
    else:
        summary_df = pd.DataFrame(summary)
    summary_df.to_csv(summary_path, index=False)




if __name__ == '__main__':
    #PER RUNNARE -> cartella repository in terminale, poi: python src/train.py
    main()
