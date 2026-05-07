import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import GINEConv, global_mean_pool
from feature_tokenizer import DualFeatureTokenizer


class BaselineGNN(pl.LightningModule):
    """
    Baseline: due GNN indipendenti (proteina e ligando) senza comunicazione.
    Architettura:
        DualFeatureTokenizer → GINEConv -> mean pool -> MLP -> pAffinity
    """
    
    def __init__(self, token_dim=32, node_embedding_dim=128, edge_embedding_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters() #salva tutti gli hyperparam passati al costruttore, accessibili via self.hparams

        self.tokenizer = DualFeatureTokenizer(token_dim, node_embedding_dim, edge_embedding_dim)
        self.prot_gnn = GINEConv(
            nn.Sequential(
                nn.Linear(node_embedding_dim, node_embedding_dim),
                nn.ReLU(),
                nn.Linear(node_embedding_dim, node_embedding_dim),
            ), edge_dim=edge_embedding_dim
        )
        self.lig_gnn  = GINEConv(
            nn.Sequential(
                nn.Linear(node_embedding_dim, node_embedding_dim),
                nn.ReLU(),
                nn.Linear(node_embedding_dim, node_embedding_dim),
            ), edge_dim=edge_embedding_dim
        )
        self.mlp= nn.Sequential(
            nn.Linear(2 * node_embedding_dim, node_embedding_dim),
            nn.ReLU(),
            nn.Linear(node_embedding_dim, 1),
        )
        
    def forward(self, prot_batch, lig_batch):
        prot, lig = self.tokenizer(prot_batch, lig_batch)

        h_p = self.prot_gnn(prot.x, prot.edge_index, prot.edge_attr)
        h_l = self.lig_gnn(lig.x, lig.edge_index, lig.edge_attr)
        
        #vettore di embedding del grafo: media dei nodi (global mean pool) | h [N, emb_dim] -> h [batch_size, emb_dim]
        h_p = global_mean_pool(h_p, prot.batch)
        h_l = global_mean_pool(h_l, lig.batch)

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

        