import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import GINEConv, global_mean_pool
from torchmetrics import PearsonCorrCoef
from features_embedding import DualGraphTokenizer


class BaselineGNN(pl.LightningModule):
    """
    Baseline: due GNN indipendenti (proteina e ligando) senza comunicazione.
    Architettura:
        DualGraphTokenizer → GINEConv -> mean pool -> MLP -> pAffinity
    """
    
    def __init__(self, token_dim=32, node_embedding_dims=[128], edge_embedding_dims=[64], num_layers=1, lr=1e-3, normalization=True, dropout_p=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.val_pearson = PearsonCorrCoef()

        self.node_embedding_dim = node_embedding_dims[-1]
        self.edge_embedding_dim = edge_embedding_dims[-1]

        self.tokenizer = DualGraphTokenizer(
            token_dim, 
            node_ligand_hidden_sizes=node_embedding_dims, 
            edge_ligand_hidden_sizes=edge_embedding_dims,
            node_protein_hidden_sizes=node_embedding_dims,
            edge_protein_hidden_sizes=edge_embedding_dims
        )
        self.prot_gnn_layers = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
                    nn.ReLU(),
                    nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
                ), edge_dim=self.edge_embedding_dim
            ) for _ in range(num_layers) 
        ])
        
        self.lig_gnn_layers  = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
                    nn.ReLU(),
                    nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
                ), edge_dim=self.edge_embedding_dim
            ) for _ in range(num_layers)
        ])
        
        mlp_layers = [nn.Linear(2 * self.node_embedding_dim, self.node_embedding_dim)]
        if normalization:
            mlp_layers.append(nn.LayerNorm(self.node_embedding_dim))
        mlp_layers += [
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.node_embedding_dim, 1)
        ]
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, prot_batch, lig_batch):
        prot, lig = self.tokenizer(prot_batch, lig_batch)

        for layer in self.prot_gnn_layers:
            prot.x = layer(prot.x, prot.edge_index, prot.edge_attr)
        for layer in self.lig_gnn_layers:
            lig.x = layer(lig.x, lig.edge_index, lig.edge_attr)
       
        # pool) | h [N, emb_dim] -> h [batch_size, emb_dim]
        h_p = global_mean_pool(prot.x, prot.batch)
        h_l = global_mean_pool(lig.x, lig.batch)

        return self.mlp(torch.cat([h_p, h_l], dim=-1)).squeeze(-1)
    
    def _step(self, batch, stage):
        prot, lig = batch
        y = prot.y.squeeze(-1)
        y_hat = self(prot, lig)
        loss = nn.functional.mse_loss(y_hat, y)
        mae  = nn.functional.l1_loss(y_hat, y)
        self.log(f"{stage}_loss", loss, batch_size=y.size(0), prog_bar=True)
        self.log(f"{stage}_mae",  mae,  batch_size=y.size(0), prog_bar=False)
        if stage == "val":
            self.val_pearson.update(y_hat.detach(), y.detach())
        return loss

    def on_validation_epoch_end(self):
        self.log("val_pearson_r", self.val_pearson.compute(), prog_bar=True)
        self.val_pearson.reset()

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        