import optuna
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from utils import ComplexDataModule
from graphmatchingnetwork import GraphMatchingNetwork

LOSS = 'val_loss_combined'

def objective(trial):

    token_dim = trial.suggest_categorical('token_dim', [16, 32, 64])
    node_dim_str = trial.suggest_categorical(
        'node_embedding_dims',
        ["32,16", "64,32", "32,32", "64,32,16","64,24,24", "32"]
    )
    edge_dim_str = trial.suggest_categorical(
        'edge_embedding_dims',
        ["32,16", "64,32", "32,32", "64,32,16", "64,24,24", "32"]
    )
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout_p = trial.suggest_float("dropout_p", 0.0, 0.4)
    normalization = trial.suggest_categorical("normalization", [True,False])
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    l2_norm = trial.suggest_float('l2_norm', 0.0, 1e-3)
    max_lr = trial.suggest_float('max_lr', 1e-4, 1e-3, log=True)
    min_lr = trial.suggest_float('min_lr', 1e-7, 1e-5, log=True)
    warmup_step_percentage = trial.suggest_float('warmup_percentage', 0.05, 0.2)

    node_embedding_dims = [int(x) for x in node_dim_str.split(",")]
    edge_embedding_dims = [int(x) for x in edge_dim_str.split(",")]

    print("\n" + "="*50)
    print(f"TRIAL {trial.number}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")

    data_module = ComplexDataModule(
        processed_dir="data/processed",
        csv_path="data/raw/PL_cleaned_index.csv",
        batch_size=batch_size,
        limit=2000,
        seed=47,
    )

    model = GraphMatchingNetwork(
        token_dim=token_dim,
        node_embedding_dims=node_embedding_dims,
        edge_embedding_dims=edge_embedding_dims,
        num_layers=num_layers,
        lr=1e-3,
        max_lr=max_lr,
        min_lr=min_lr,
        l2_norm=l2_norm,
        warmup_step_percentage=warmup_step_percentage,
        use_dynamic_schedule=True,
        dropout_p=dropout_p,
        normalization=normalization,
    )

    trainer = pl.Trainer(
        max_epochs=500,
        precision="bf16-mixed",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        callbacks=[
            EarlyStopping(monitor=LOSS, patience=15, mode='min'),
            PyTorchLightningPruningCallback(trial, monitor=LOSS),
        ],
    )

    trainer.fit(model, datamodule=data_module)
    return trainer.callback_metrics[LOSS].item()


if __name__ == "__main__":

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=15,
        n_warmup_steps=30,
        interval_steps=5,
    )
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="gnn_hyperparam_search",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=400, timeout=9000)

    print(f"{study.study_name} - Best HP found:")
    print(study.best_params)
    print(f"Best {LOSS}: {study.best_value:.4f}")
    