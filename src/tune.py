import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# Importa i tuoi moduli
from src.complex_dataset import ComplexDataModule
from src.graphmatchingnetwork import GraphMatchingNetwork

LOSS='val_loss_combined'
data_module = ComplexDataModule(
    processed_dir="data/processed",
    csv_path="data/raw/PL_cleaned_index.csv",
    batch_size=16,
    limit=1000,
    seed=47,
)
data_module.setup()

def objective(trial):
    
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    token_dim = trial.suggest_categorical('token_dim', [8, 16, 32, 64])
    node_embedding_dims = trial.suggest_categorical('node_embedding_dims', [(32, 16), (64, 32), (128, 64), (32,), (48,), (64,)])
    edge_embedding_dims = trial.suggest_categorical('edge_embedding_dims', [(32, 16), (64, 32), (128, 64), (32,), (48,), (64,)])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout_p      = trial.suggest_float("dropout_p", 0.0, 0.4)
    normalization  = trial.suggest_categorical("normalization", [True])
    
    
    model = GraphMatchingNetwork(
        token_dim=token_dim,
        node_embedding_dims=list(node_embedding_dims),
        edge_embedding_dims=list(edge_embedding_dims),
        num_layers=num_layers,
        lr=lr,
        dropout_p=dropout_p,
        normalization=normalization,
    )
    
    trainer = pl.Trainer(
        max_epochs=200,
        precision="bf16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[
            EarlyStopping(monitor=LOSS, patience=15, mode='min'),
            PyTorchLightningPruningCallback(trial, monitor=LOSS)
        ],
    )
    
    trainer.fit(model, datamodule=data_module)
    return trainer.callback_metrics[LOSS].item()


if __name__ == "__main__":
    
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, #fai finire almeno 5 trial prima di iniziare a potare
        n_warmup_steps=20, #dal momento che il trial è iniziato, aspetta almeno 14 epoche prima di potare
        interval_steps=5 #ogni 5 step, controlla se il trial può essere potato
    )
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="gnn_hyperparam_search",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=400, timeout=9000)  # E
    
    print(f"{study.study_name} - Best HP found:")
    print(study.best_params)
    print(f"Best {LOSS}: {study.best_value:.4f}")
    