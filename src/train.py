import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from complex_dataset import ComplexDataset, complex_collate_fn, build_complex_dataset
from baseline import BaselineGNN
from graphmatchingnetwork import GraphMatchingNetwork


def main(config_path='config.yaml'):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    processed_dir = config['dataset']['processed_dir']
    csv_path      = config['dataset']['PL_csv']
    batch_size    = config['training']['batch_size']

    full_dataset = build_complex_dataset(processed_dir, csv_path, limit=1000)
    all_codes    = full_dataset.pdb_codes

    # Split train/val per codice PDB
    train_codes, val_codes = train_test_split(all_codes, test_size=0.2, random_state=47)

    train_ds = ComplexDataset(processed_dir, train_codes)
    val_ds   = ComplexDataset(processed_dir, val_codes)

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=complex_collate_fn, 
        num_workers=4,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=complex_collate_fn,
        num_workers=0,
    )

    print(f"Train: {len(train_ds)} complessi | Val: {len(val_ds)} complessi")
    """
        model = BaselineGNN(
            token_dim=config['model']['token_dim'],
            node_embedding_dim=config['model']['node_embedding_dim'],
            edge_embedding_dim=config['model']['edge_embedding_dim'],
            lr=config['model']['lr'],
        )
    """
    model = GraphMatchingNetwork( # INVECE DI BaselineGNN
    token_dim=config['model']['token_dim'],
    node_embedding_dim=config['model']['node_embedding_dim'],
    edge_embedding_dim=config['model']['edge_embedding_dim'],
    num_layers=3, # Puoi regolarlo da config se vuoi
    lr=config['model']['lr'],
)

    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        precision='16-mixed',
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
