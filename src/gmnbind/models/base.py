import pytorch_lightning as pl
from gmnbind.metrics import all_metrics
from gmnbind.models.tokenizer import DualGraphTokenizer
import torch
import torch.nn as nn
import math

class BaseAffinityModel(pl.LightningModule):
    def __init__(self, token_dim=32, node_embedding_dims=[128], edge_embedding_dims=[64],
                 num_layers=2, dropout_p=0.1, normalization=True,
                 lr=1e-3, max_lr=1e-3, min_lr=1e-6, l2_norm=0.0,
                 warmup_step_percentage=0.1, use_dynamic_schedule=False):
        
        super().__init__()
        self.save_hyperparameters()
        
        # predictions/targets accumulated over an epoch, per stage
        
        self._epoch_outputs = {"val": [], "test": []}

        self.node_embedding_dim = node_embedding_dims[-1]
        self.edge_embedding_dim = edge_embedding_dims[-1]

        self.tokenizer = DualGraphTokenizer(
            token_dim,
            node_hidden_sizes=node_embedding_dims,
            edge_ligand_hidden_sizes=edge_embedding_dims,
            edge_protein_hidden_sizes=edge_embedding_dims,
        )
        
    # the prediction head MLP is built by the subclass with _build_head(), used in both models  
    def _build_head(self):
        d = self.node_embedding_dim
        layers = [nn.Linear(2 * d, d)]
        if self.hparams.normalization:
            layers.append(nn.LayerNorm(d))
        layers += [
            nn.ReLU(), 
            nn.Dropout(self.hparams.dropout_p), 
            nn.Linear(d, 1)
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, prot_batch, lig_batch):
        raise NotImplementedError
    
    def _step(self, batch, stage):
        prot, lig = batch
        y = prot.y.squeeze(-1)
        y_hat = self(prot, lig)
        loss = nn.functional.mse_loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        self.log(f"{stage}_loss", loss, batch_size=y.size(0), prog_bar=True)
        if stage == "train":
            self.log("train_mae", mae, batch_size=y.size(0), prog_bar=False)
        
        if stage in self._epoch_outputs:
            self._epoch_outputs[stage].append(
                (y_hat.detach().float().cpu(), y.detach().float().cpu())
            )
        return loss
                
    def _log_epoch_metrics(self, stage):
        outputs = self._epoch_outputs[stage]
        if not outputs:
            return
        y_hat = torch.cat([o[0] for o in outputs])
        y     = torch.cat([o[1] for o in outputs])
        for name, value in all_metrics(y, y_hat).items():
            self.log(f"{stage}_{name}", value, prog_bar=(name == "pearson"))
        outputs.clear()
    
        
    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def test_step(self, batch, _):
        return self._step(batch, "test")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")
            
    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")

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
