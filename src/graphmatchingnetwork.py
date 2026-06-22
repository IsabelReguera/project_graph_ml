import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import global_mean_pool

from features_embedding import DualGraphTokenizer
from utils import DATASET_SCHEMAS
from gmn_layers import GMNLayer

class GraphMatchingNetwork(pl.LightningModule):
    """
    Modello GMN completo.
    I grafi di proteina e ligando comunicano attivamente in ogni layer prima di essere compressi.
    Architettura:
        DualFeatureTokenizer → [GMNLayer x num_layers] → mean pool → MLP → pAffinity
    
    """
    def __init__(self, token_dim=32, node_embedding_dims=[128], edge_embedding_dims=[64],
                 num_layers=2, lr=1e-3, dropout_p=0.1, normalization=True):
        super().__init__()
        self.save_hyperparameters()
        
        self.node_embedding_dim = node_embedding_dims[-1]
        self.edge_embedding_dim = edge_embedding_dims[-1]   

        self.tokenizer = DualGraphTokenizer(
            token_dim=token_dim,
            node_ligand_hidden_sizes=node_embedding_dims,
            edge_ligand_hidden_sizes=edge_embedding_dims,
            node_protein_hidden_sizes=node_embedding_dims,
            edge_protein_hidden_sizes=edge_embedding_dims
        )

        self.gmn_layers = nn.ModuleList([
            GMNLayer(self.node_embedding_dim, self.edge_embedding_dim,
                     dropout_p=dropout_p, use_norm=normalization)
            for _ in range(num_layers)
        ])

        # Identico al prediction MLP del Baseline per confronto equo
        mlp_layers = [nn.Linear(2 * self.node_embedding_dim, self.node_embedding_dim)]
        if normalization:
            mlp_layers.append(nn.LayerNorm(self.node_embedding_dim))
        mlp_layers += [
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.node_embedding_dim, 1),
        ]
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, prot_batch, lig_batch):
        #Proiezione nello spazio comune
        prot, lig = self.tokenizer(prot_batch, lig_batch)

        #Propagazione GMN (Loop sui layer)
        # Ad ogni passo, proteina e ligando si parlano e si aggiornano
        for layer in self.gmn_layers:
            prot.x, lig.x = layer(
                prot.x, prot.edge_index, prot.edge_attr, prot.batch,
                lig.x, lig.edge_index, lig.edge_attr, lig.batch
            )
        
        #Readout (Media finale)
        h_p = global_mean_pool(prot.x, prot.batch)
        h_l = global_mean_pool(lig.x, lig.batch)

        #Predizione
        return self.mlp(torch.cat([h_p, h_l], dim=-1)).squeeze(-1)
    
    
    def _step(self, batch, stage):
        prot, lig = batch
        y = prot.y.squeeze(-1)
        y_hat = self(prot, lig)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log(f"{stage}_loss", loss, batch_size=y.size(0), prog_bar=True)
        return loss
        
    def training_step(self, batch, _):
        return self._step(batch, "train")
    
    def validation_step(self, batch, _):
        return self._step(batch, "val")
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)