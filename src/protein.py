import os
import networkx as nx
from functools import partial

import torch
import cpdb
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold
from graphein.protein.visualisation import plotly_protein_structure_graph
import py3Dmol

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
            pdb_code = os.path.basename(path).split('/')[-1].split('_')[0]  # Estrae il codice PDB dal nome del file
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
        edge_funcs = [
            partial(add_distance_threshold, long_interaction_threshold=0, threshold=8.0)
        ]
        config = ProteinGraphConfig(edge_construction_functions=edge_funcs)

        if self.pdb_path:
            #print(f"Costruzione grafo dal file locale: {self.pdb_path}...")
            # Workaround: cpdb.parse_pdb_file (estensione C) crasha non so perchè;
            # leggiamo il file come stringa e usiamo il parser Python.
            with open(self.pdb_path, "r") as f:
                raw_df = cpdb.parse(pdb_str=f.read())
            self._graph = construct_graph(config=config, df=raw_df)

        else:
            #print(f"Download e costruzione grafo per PDB: {self.pdb_code}...")
            self._graph = construct_graph(config=config, pdb_code=self.pdb_code)

    @property
    def graph(self):
        """Proprietà per accedere al grafo in modo trasparente."""
        self._ensure_graph()
        return self._graph

    def get_info(self):
        """info sul grafo - test"""
        g = self.graph
        return {
            "n_nodes": len(g.nodes()),
            "n_edges": len(g.edges()),
            "nodes_sample": list(g.nodes())[:5]
        }

    def _get_node_features(self):
        """Feature residui:
            Meiler embedding 7D -> rappresentazione continua dell'amminoacido
            b_factor           -> flessibilità/mobilità del residuo
        Shape: [N, 8], float32
        """
        feats = []
        for _, d in self.graph.nodes(data=True):
            meiler = list(d["meiler"].values)        # 7 valori float
            feats.append(meiler + [d["b_factor"]])
        return torch.tensor(feats, dtype=torch.float32)

    def _get_edge_index(self):
        """Edge index bidirezionale"""
        src, dst = [], []
        nodes = list(self.graph.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        for u, v in self.graph.edges():
            src.extend([idx[u], idx[v]])
            dst.extend([idx[v], idx[u]])
        return torch.tensor([src, dst], dtype=torch.long)

    def _get_edge_features(self):
        """Feature arco (bidirezionali):
            distanza Ca-Ca in Angstrom
        """
        feats = []
        for _, _, d in self.graph.edges(data=True):
            f = [d["distance"]]
            feats.extend([f, f])
        return torch.tensor(feats, dtype=torch.float32)

    def _get_3d_coordinates(self):
        """Coordinate Ca. Shape: [N, 3], float32."""
        pos = [d["coords"].tolist() for _, d in self.graph.nodes(data=True)]
        return torch.tensor(pos, dtype=torch.float32)

    def get_pyg_graph(self):
        """Restituisce (x, edge_index, edge_attr, pos) pronti per PyTorch Geometric."""
        x          = self._get_node_features()
        edge_index = self._get_edge_index()
        edge_attr  = self._get_edge_features()
        pos        = self._get_3d_coordinates()
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


# Esempio di utilizzo
if __name__ == "__main__":
    
    prot = Protein(pdb_code="3pd9")
    print(f"Informazioni proteina: {prot.get_info()}")
    print(f"Il grafo ha {len(prot.graph.nodes())} nodi.")