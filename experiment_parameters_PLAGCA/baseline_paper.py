"""
Variant of BaselineGNN with a parametric MLP head.

Identical to src/baseline.py in every other respect (same two independent
GINEConv branches, same tokenizer dispatch, same mean-pool readout).
The only addition is the `mlp_hidden_sizes` constructor argument, which
lets us replicate the PLAGCA MLP head [512, 256, 128, 1] instead of the
default 2-layer head used by the original model.

Kept in experiment_parameters_PLAGCA/ to avoid touching src/baseline.py.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import GINEConv, global_mean_pool
from torchmetrics import PearsonCorrCoef

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features_embedding import DualGraphTokenizer


def build_mlp_head(input_dim, hidden_sizes, dropout_p, normalization):
    """
    Build an MLP head whose last element in `hidden_sizes` is the output dim.
    - LayerNorm + ReLU + Dropout after every Linear except the last.
    - No dropout / activation / norm after the output Linear.

    Example: input_dim=256, hidden_sizes=[512, 256, 128, 1] produces
        Linear(256 -> 512) -> LN -> ReLU -> Dropout
        Linear(512 -> 256) -> LN -> ReLU -> Dropout
        Linear(256 -> 128) -> LN -> ReLU -> Dropout
        Linear(128 -> 1)
    """
    layers = []
    prev = input_dim
    for i, h in enumerate(hidden_sizes):
        layers.append(nn.Linear(prev, h))
        is_last = (i == len(hidden_sizes) - 1)
        if not is_last:
            if normalization:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
        prev = h
    return nn.Sequential(*layers)


class BaselinePaperGNN(pl.LightningModule):
    """
    Baseline con MLP head parametrico. Compatibile con il training loop
    dello script di ablation.
    """

    def __init__(self, token_dim=32, node_embedding_dims=[128], edge_embedding_dims=[64],
                 num_layers=2, dropout_p=0.1, normalization=True,
                 lr=1e-3, max_lr=1e-3, min_lr=1e-6, l2_norm=0.0,
                 warmup_step_percentage=0.1, use_dynamic_schedule=True,
                 embedding_type="tokenizer", cat_emb_dim=8,
                 mlp_hidden_sizes=None,
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
            edge_protein_hidden_sizes=edge_embedding_dims,
            embedding_type=embedding_type,
            cat_emb_dim=cat_emb_dim,
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

        self.lig_gnn_layers = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
                    nn.ReLU(),
                    nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
                ), edge_dim=self.edge_embedding_dim
            ) for _ in range(num_layers)
        ])

        mlp_input = 2 * self.node_embedding_dim
        if mlp_hidden_sizes is None:
            mlp_hidden_sizes = [self.node_embedding_dim, 1]
        self.mlp = build_mlp_head(mlp_input, mlp_hidden_sizes, dropout_p, normalization)

    def forward(self, prot_batch, lig_batch):
        prot, lig = self.tokenizer(prot_batch, lig_batch)
        for layer in self.prot_gnn_layers:
            prot.x = layer(prot.x, prot.edge_index, prot.edge_attr)
        for layer in self.lig_gnn_layers:
            lig.x = layer(lig.x, lig.edge_index, lig.edge_attr)
        h_p = global_mean_pool(prot.x, prot.batch)
        h_l = global_mean_pool(lig.x, lig.batch)
        return self.mlp(torch.cat([h_p, h_l], dim=-1)).squeeze(-1)

    def _step(self, batch, stage):
        prot, lig = batch
        y = prot.y.squeeze(-1)
        y_hat = self(prot, lig)
        loss = nn.functional.mse_loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        self.log(f"{stage}_loss", loss, batch_size=y.size(0), prog_bar=True)
        self.log(f"{stage}_mae", mae, batch_size=y.size(0), prog_bar=False)
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
        if self.hparams.use_dynamic_schedule:
            max_lr = self.hparams.max_lr
            min_lr = self.hparams.min_lr
            warmup_percentage = self.hparams.warmup_step_percentage
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=max_lr,
                betas=(0.9, 0.999), eps=1e-8,
                weight_decay=self.hparams.l2_norm,
            )
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(total_steps * warmup_percentage)
            min_lr_ratio = min_lr / max_lr
            decay_power = 2.0

            def cosine_with_warmup_lambda(step):
                step = max(1, step)
                if step <= warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                base_cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                steep = base_cosine ** decay_power
                return min_lr_ratio + (1.0 - min_lr_ratio) * steep

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        else:
            return torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr,
                weight_decay=self.hparams.l2_norm,
            )
