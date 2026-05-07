import os
import networkx as nx
from functools import partial

import torch
import cpdb

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_peptide_bonds, add_distance_threshold
from graphein.protein.features.nodes.amino_acid import meiler_embedding
from graphein.protein.visualisation import plotly_protein_structure_graph
import py3Dmol
# funzioni di annotazione nodi 
node_funcs = [
    meiler_embedding,      # Carica il Meiler Embedding, proprietà chimico-fisiche per ogni aminoacido
]

# funzioni di costruzione degli archi
edge_funcs = [
    add_peptide_bonds,    # Archi (legami covalenti)
    partial(add_distance_threshold,  long_interaction_threshold=0, threshold=8.0) # Archi spaziali
]

# configurazione del grafo con le funzioni specificate
GRAPH_CONFIG = ProteinGraphConfig(
    node_metadata_functions=node_funcs,
    edge_construction_functions=edge_funcs,
    verbose=False
)
AMMINO_ACIDS_DICT = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10,'LYS': 11,'MET': 12,'PHE': 13,'PRO': 14,
    'SER': 15,'THR': 16,'TRP': 17,'TYR': 18,'VAL': 19
}

class Protein:
    def __init__(self, pdb_code, pdb_path=None):
        self.pdb_code = pdb_code
        self.pdb_path = pdb_path
        self._graph = None

    @classmethod
    def from_pdb_path(cls, path):
        """Crea un'istanza di Protein a partire da un file PDB locale."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File non trovato: {path}")
        else:
            pdb_code = os.path.basename(path).split('_')[0]  # Estrae il codice PDB dal nome del file
        return cls(pdb_code=pdb_code, pdb_path=path)

    @classmethod
    def from_pdb_code(cls, pdb_code):
        """Crea un'istanza di Protein scaricando la struttura da RCSB."""
        if not pdb_code or len(pdb_code) != 4:
            raise ValueError(f"PDB code non valido: {pdb_code}")
        return cls(pdb_code=pdb_code)

    def _ensure_graph(self):
        """Genera il grafo lazy alla prima lettura."""
        if self._graph is not None:
            return
        
        config = GRAPH_CONFIG

        if self.pdb_path:
            # Workaround: construct_graph(config=config, path=self.pdb_path) crasha non so perchè;
            # leggiamo il file come stringa e usiamo il parser Python.
            with open(self.pdb_path, "r") as f:
                raw_df = cpdb.parse(pdb_str=f.read())
            self._graph = construct_graph(config=config, df=raw_df)

        else:
            self._graph = construct_graph(config=config, pdb_code=self.pdb_code)

    @property
    def graph(self):
        """Proprietà per accedere al grafo in modo trasparente."""
        self._ensure_graph()
        return self._graph

    def _get_node_features(self):#TODO cerca feature foglietto beta
        """Feature residui:
            Meiler embedding 7D  -> rappresentazione chimico-fisica dell'amminoacido. (idrofobicità, volume sterico, Polarizzabilità, polarità, Potenziale Isuelettrico, Indice di Refrattività, Propensione per l'Elica alpha) 
            b_factor             -> flessibilità/mobilità dell'amminoacido (misura l'incertezza della posizione di un atomo).
            res_name             -> tipo di amminoacido ex: ALA, ARG, ecc. (potrebbe essere codificato come one-hot ma trppo sparso)
        """
        feats = []
        for _, d in self.graph.nodes(data=True):
            meiler = list(d["meiler"].values)       # 7 valori 
            b_factor = d["b_factor"]                # 1 valore 
            res_name = AMMINO_ACIDS_DICT.get(d["residue_name"], 20)  # codice numerico dell'amminoacido (20 = sconosciuto)
            feats.append(meiler + [b_factor] + [res_name])  # concatenazione delle feature
        return torch.tensor(feats, dtype=torch.float32)

    def _get_edges_and_features(self):
        """
        Feature arco (bidirezionali):
            0: Tipo legame (0.0 = contatto spaziale 1.0 = peptide bond covalente, ... altri tipi in futuro)
            1: Distanza Ca-Ca in Å
        """
        nodes = list(self.graph.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        src, dst, edge_attrs = [], [], []

        for u, v, d in self.graph.edges(data=True):
            is_peptide = "peptide_bond" in d.get("kind", set())
            bond_type = 1.0 if is_peptide else 0.0
            distance  = d["distance"]
            attr = [bond_type, distance]

            src.extend([idx[u], idx[v]])
            dst.extend([idx[v], idx[u]])
            edge_attrs.extend([attr, attr])

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attrs, dtype=torch.float32)
        return edge_index, edge_attr

    def _get_3d_coordinates(self):
        """Coordinate ?? non so se serve"""
        pos = [d["coords"].tolist() for _, d in self.graph.nodes(data=True)]
        return torch.tensor(pos, dtype=torch.float32)

    def get_pyg_graph(self):
        """Restituisce (x, edge_index, edge_attr, pos) pronti per PyTorch Geometric."""
        x                    = self._get_node_features()
        pos                  = self._get_3d_coordinates()
        edge_index, edge_attr = self._get_edges_and_features()
        return x, edge_index, edge_attr, pos
    
    def visualize_protein_graph(self):
        """Visualizza la struttura della proteina come grafo 3D usando Plotly."""
        g = self.graph
    
        fig = plotly_protein_structure_graph(
            g,
            colour_edges_by="kind",
            colour_nodes_by="residue_name",  # Colora per tipo di amminoacido
            node_size_multiplier=0.4,       # dimensione dei nodi
            node_alpha=0.7,               # trasparenza dei nodi
            plot_title=f"Graph: {self.pdb_code}"
        )

        #stile delle linee per i diversi tipi di interazione
        fig.update_traces(line=dict(width=3), selector=dict(type='scatter3d'))

        # Miglioramento estetico del Layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data' # Mantiene le proporzioni reali della proteina
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig.show()
    
    def visualize_protein_3D(self):
        view = py3Dmol.view(width=800, height=600)

        if self.pdb_path and os.path.exists(self.pdb_path):
            # Caricamento da file 
            with open(self.pdb_path, 'r') as f:
                pdb_data = f.read()
            view.addModel(pdb_data, 'pdb')
            print(f"Visualizzazione da file locale: {self.pdb_path}")
        elif self.pdb_code:
            # Download
            view = py3Dmol.view(query=f'pdb:{self.pdb_code}')
            print(f"Visualizzazione da codice PDB: {self.pdb_code}")
        else:
            raise ValueError("Nessun codice o percorso file disponibile per la visualizzazione.")

        # Stile e impostazioni
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        # aggiungi le sidechains per i residui vicini se serve
        view.addStyle({'resn': 'LIG'}, {'stick': {}}) 
        
        view.zoomTo()
        return view.show()


if __name__ == "__main__":
    prot = Protein.from_pdb_code("3pd9")
    x, edge_index, edge_attr, pos = prot.get_pyg_graph()
    print(f"Nodi: {x.shape}  |  Archi: {edge_index.shape}  |  Edge attr: {edge_attr.shape}  |  Pos: {pos.shape}")