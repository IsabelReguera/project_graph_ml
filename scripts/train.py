"""
Train loop for GMN and Baseline models using PyTorch Lightning.
This script reads configuration from a YAML file, sets up the data module, initializes the model,
 and starts the training process.
 
 Usage:
    python scripts/train.py --config configs/gmn.yaml

"""


import argparse
import os
import yaml
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
from pathlib import Path

from gmnbind.models.baseline import BaselineGNN
from gmnbind.models.gmn import GraphMatchingNetwork
from gmnbind.data.datamodule import ComplexDataModule

#to improve performance on some GPUs
torch.set_float32_matmul_precision('medium')

REPO_ROOT = Path(__file__).resolve().parents[1]   

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gmn.yaml")
    args = parser.parse_args()
    config_path = REPO_ROOT/args.config

    # --- Load configuration from YAML file --------
    with open(config_path) as f:
        config = yaml.safe_load(f)

    processed_dir   = REPO_ROOT/config['data']['processed_dir']
    log_dir         = REPO_ROOT/config['data']['log_dir']
    csv_path        = REPO_ROOT/config['data']['csv_path']
    
    batch_size      = config['training']['batch_size']
    model_type      = config['model'].get('model_type', 'baseline')  # Default to 'baseline' if not specified
    experiment_name = config['model']['experiment_name']
    
    token_dim = config['model']['token_dim']
    node_embedding_dims = config['model']['node_embedding_dims']
    edge_embedding_dims = config['model']['edge_embedding_dims']
    num_layers = config['model']['num_layers']
    lr=config['model']['lr']
    max_lr = config['model']['max_lr']
    min_lr = config['model']['min_lr']
    warmup_percentage=  config['model']['warmup_percentage']
    l2_norm = config['model']['l2_norm']
    dropout_p = config['model']['dropout_p']
    normalization = config['model']['normalization']
    use_dynamic_schedule = config['model']['use_dynamic_schedule']


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
        LearningRateMonitor(logging_interval='step')
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
            num_layers=num_layers,
            lr=lr,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_step_percentage=warmup_percentage,
            l2_norm=l2_norm,
            dropout_p=dropout_p,
            normalization=normalization,
            use_dynamic_schedule=use_dynamic_schedule,
        )
    else:
        model = GraphMatchingNetwork(
            token_dim=token_dim,
            node_embedding_dims=node_embedding_dims,
            edge_embedding_dims=edge_embedding_dims,
            num_layers=num_layers,
            lr=lr,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_step_percentage=warmup_percentage,
            l2_norm=l2_norm,
            dropout_p=dropout_p,
            normalization=normalization,
            use_dynamic_schedule=use_dynamic_schedule,
        )

    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        precision='bf16-mixed',
        log_every_n_steps=10,
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
        'node_embedding_dims':  node_embedding_dims[-1],
        'edge_embedding_dims':  edge_embedding_dims[-1],
        'num_layers':           num_layers,
        'normalization':        normalization,
        'dropout_p':            dropout_p,
        'l2_norm':              l2_norm,
        'lr':                   lr,
        'dynamic_schedule':     use_dynamic_schedule,
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
    main()
