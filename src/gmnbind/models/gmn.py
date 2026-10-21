import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import global_mean_pool
from torchmetrics import PearsonCorrCoef

from gmnbind.models.tokenizer import DualGraphTokenizer
from scripts.utils import DATASET_SCHEMAS
from gmnbind.models.layers import GMNLayer

class GraphMatchingNetwork(pl.LightningModule):
    """
    Modello GMN completo.
    I grafi di proteina e ligando comunicano attivamente in ogni layer prima di essere compressi.
    Architettura:
        DualFeatureTokenizer → [GMNLayer x num_layers] → mean pool → MLP → pAffinity
    
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
            token_dim=token_dim,
            node_hidden_sizes=node_embedding_dims,
            edge_ligand_hidden_sizes=edge_embedding_dims,
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
        
    def forward(self, prot_batch, lig_batch, return_attention=False):
        #Proiezione nello spazio comune
        prot, lig = self.tokenizer(prot_batch, lig_batch)

        # Propagazione GMN (Loop sui layer)
        # Se return_attention=True, teniamo le matrici dell'ultimo layer
        attention_maps = None
        for layer in self.gmn_layers:
            layer_out = layer(
                prot.x, prot.edge_index, prot.edge_attr, prot.batch,
                lig.x, lig.edge_index, lig.edge_attr, lig.batch,
                return_attention=return_attention,
            )
            if return_attention:
                prot.x, lig.x, attn_p_to_l, attn_l_to_p = layer_out
                attention_maps = (attn_p_to_l, attn_l_to_p)
            else:
                prot.x, lig.x = layer_out

        #Readout (Media finale)
        h_p = global_mean_pool(prot.x, prot.batch)
        h_l = global_mean_pool(lig.x, lig.batch)

        #Predizione
        out = self.mlp(torch.cat([h_p, h_l], dim=-1)).squeeze(-1)
        if return_attention:
            return out, attention_maps
        return out
    
    
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
            
            decay_power = 2.0
            def cosine_with_warmup_lambda(step):
                step = max(1, step)

                # Phase 1: linear warmup from ~0 to max_lr
                if step <= warmup_steps:
                    return step / warmup_steps

                # Phase 2: cosine decay from max_lr down to min_lr
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                base_cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                steep_cosine = base_cosine ** decay_power
                return min_lr_ratio + (1.0 - min_lr_ratio) * steep_cosine

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
