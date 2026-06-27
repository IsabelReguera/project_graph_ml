import torch
import math
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
    
    def __init__(self, token_dim=32, node_embedding_dims=[128], edge_embedding_dims=[64],
                 num_layers=2,  dropout_p=0.1, normalization=True,
                 lr=1e-3, max_lr=1e-3, min_lr=1e-6, l2_norm=0.0, warmup_step_percentage=0.1, use_dynamic_schedule=True
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.val_pearson = PearsonCorrCoef()

        self.node_embedding_dim = node_embedding_dims[-1]
        self.edge_embedding_dim = edge_embedding_dims[-1]

        self.tokenizer = DualGraphTokenizer(
            token_dim, 
            node_hidden_sizes=node_embedding_dims, 
            edge_ligand_hidden_sizes=edge_embedding_dims,
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
        
        mse = nn.functional.mse_loss(y_hat, y)
        mae  = nn.functional.l1_loss(y_hat, y)
        
        self.log(f"{stage}_loss", mse, batch_size=y.size(0), prog_bar=True)
        self.log(f"{stage}_mae",  mae,  batch_size=y.size(0), prog_bar=False)
        if stage == "val":
            self.val_pearson.update(y_hat.detach(), y.detach())
        return mse

    def on_validation_epoch_end(self):
        #combined loss = mse + λ * (1 - pearson_r) - for optuna
        r = self.val_pearson.compute()
        if torch.isnan(r):
            r = torch.tensor(0.0, device=self.device)
        combined_loss = 0.8* self.trainer.callback_metrics["val_loss"] + 0.2 * (1 - r)
        self.log("val_pearson_r",      r,              prog_bar=True)
        self.log("val_loss_combined",  combined_loss,  prog_bar=False)
        self.val_pearson.reset()
        

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # AdamW with linear warmup + cosine decay schedule.
        # LR rises linearly for warmup_step_percentage of total steps, then decays
        # smoothly via cosine annealing down to min_lr.
        if self.hparams.use_dynamic_schedule:
            max_lr = self.hparams.max_lr
            min_lr = self.hparams.min_lr
            warmup_percentage = self.hparams.warmup_step_percentage

            # Optimizer starts at max_lr; the scheduler acts as a multiplier in [min_lr/max_lr, 1.0]
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=max_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.hparams.l2_norm
            )

            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(total_steps * warmup_percentage)
            min_lr_ratio = min_lr / max_lr

            def cosine_with_warmup_lambda(step):
                step = max(1, step)

                # Phase 1: linear warmup from ~0 to max_lr
                if step <= warmup_steps:
                    return step / warmup_steps

                # Phase 2: cosine decay from max_lr down to min_lr
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup_lambda)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }
            
        else: 
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2_norm)


        