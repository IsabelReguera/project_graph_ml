"""
Train loop for GMN and Baseline models using PyTorch Lightning.
This script reads configuration from a YAML file, trains the model with gmnbind.training.train
(best checkpoint saved, learning rate logged) and appends a row to <log_dir>/summary.csv.

 Usage:
    python scripts/train.py --config configs/gmn.yaml

"""
import warnings
warnings.filterwarnings("ignore", message=".*torch.jit.script.*is deprecated.*", category=FutureWarning)

import argparse
import os
from pathlib import Path

import pandas as pd

from gmnbind.training import load_config, train

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gmn.yaml")
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)
    log_dir = REPO_ROOT / config['data']['log_dir']
    experiment_name = config['model']['experiment_name']
    model_cfg = config['model']

    result = train(config)        # save_checkpoint=True, lr_monitor=True

    # --- LOGGING METRICS ------
    best_val = result['val_loss']
    best_epoch = result['best_epoch']
    best_val_mae = result['val_mae']
    best_val_pearson = result['val_pearson']
    print(f"[{experiment_name}]  best val_loss = {best_val:.4f} | R = {best_val_pearson:.4f} | epoch = {best_epoch} ")

    summary = [{
        'name':                 experiment_name + f"_{result['version']}",
        'model':                model_cfg.get('model_type', 'baseline'),
        'best_val_loss':        best_val,
        'best_val_mae':         best_val_mae,
        'pearson_r':            best_val_pearson,
        'node_embedding_dims':  model_cfg['node_embedding_dims'][-1],
        'edge_embedding_dims':  model_cfg['edge_embedding_dims'][-1],
        'num_layers':           model_cfg['num_layers'],
        'normalization':        model_cfg['normalization'],
        'dropout_p':            model_cfg['dropout_p'],
        'l2_norm':              model_cfg['l2_norm'],
        'lr':                   model_cfg['lr'],
        'dynamic_schedule':     model_cfg['use_dynamic_schedule'],
        'batch_size':           config['training']['batch_size'],
    }]

    # Save summary to CSV
    summary_path = os.path.join(log_dir, 'summary.csv')
    if os.path.exists(summary_path):
        existing_summary = pd.read_csv(summary_path)
        summary_df = pd.concat([existing_summary, pd.DataFrame(summary)], ignore_index=True)
    else:
        summary_df = pd.DataFrame(summary)
    summary_df.to_csv(summary_path, index=False)


if __name__ == '__main__':
    main()
