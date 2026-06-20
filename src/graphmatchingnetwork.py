import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import global_mean_pool

#from features_embedding import DualGraphTokenizer
from feature_tokenizer import DualGraphTokenizer
from utils import DATASET_SCHEMAS
from gmn_layers import GMNLayer

class GraphMatchingNetwork(pl.LightningModule):
    """
    Modello GMN completo.
    I grafi di proteina e ligando comunicano attivamente in ogni layer prima di essere compressi.
    Architettura:
        DualFeatureTokenizer → [GMNLayer x num_layers] → mean pool → MLP → pAffinity
    
    """
    def __init__(self, token_dim=32, node_embedding_dim=128, edge_embedding_dim=64, num_layers=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        """ ho messo sotto lo stessso tokenizer che usa la baseline baseline(temporaneo è corretto quello in features_embedding.py) 
            sono uguali ma questo ha un solo layer quindi 
                - node_embedding_dim e edge_embedding_dim sono interi e non liste 
                se vuoi usare il tokenizer di features_embedding.py aggirna anche quello nella baseline e 
                occhio ai parametri che vengono passati al costruttore nel train (sono liste di interi)!!
                è una cavolata ma va modificato il anche altro perchè node/edge embedding dimi viene usato da altre parti come intero,poi fixo
            ho messo questo per semplicità
        
        self.tokenizer = DualGraphTokenizer(  
            schemas_dict=DATASET_SCHEMAS,
            node_protein_hidden_sizes=[node_embedding_dim], 
            edge_protein_hidden_sizes=[edge_embedding_dim],
            node_ligand_hidden_sizes=[node_embedding_dim],
            edge_ligand_hidden_sizes=[edge_embedding_dim],
            token_dim=token_dim
        )"""
        #tokenizer baseline 
        self.tokenizer = DualGraphTokenizer(token_dim, node_embedding_dim, edge_embedding_dim)
        #mettiamo in fila num_layers (parametro) GMNLayer
        self.gmn_layers = nn.ModuleList([
            GMNLayer(node_embedding_dim, edge_embedding_dim) 
            for _ in range(num_layers)
        ])
        
        #TODO: valutare se aggiungere layer norm come nella baseline e GELU invece di ReLU
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_embedding_dim, node_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(node_embedding_dim, 1),
        )
        
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