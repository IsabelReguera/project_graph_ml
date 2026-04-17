import os
import networkx as nx
from functools import partial

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold

class Protein:
    def __init__(self, pdb_code=None, pdb_path=None):
        self.pdb_code = pdb_code
        self.pdb_path = pdb_path
        
        # Il grafo viene inizializzato a None e creato solo al bisogno
        self._graph = None

    def _ensure_graph(self):
        """Metodo privato che genera il grafo solo se non esiste già."""
        if self._graph is None:
            # Configurazione degli archi (es. soglia 8.0 Angstrom)
            edge_funcs = [
                partial(add_distance_threshold, long_interaction_threshold=0, threshold=8.0)
            ]
            config = ProteinGraphConfig(edge_construction_functions=edge_funcs)
            
            if self.pdb_path:
                if not os.path.exists(self.pdb_path):
                    raise FileNotFoundError(f"File non trovato: {self.pdb_path}")
                print(f"Costruzione grafo dal file locale: {self.pdb_path}...")
                self._graph = construct_graph(config=config, pdb_path=self.pdb_path)
            
            elif self.pdb_code:
                print(f"Download e costruzione grafo per PDB: {self.pdb_code}...")
                self._graph = construct_graph(config=config, pdb_code=self.pdb_code)
            
            else:
                raise ValueError("pdb_code non fornito.")

    @property
    def graph(self):
        """Proprietà per accedere al grafo in modo trasparente."""
        self._ensure_graph()
        return self._graph

    def get_info(self):
        """Esempio di metodo che usa il grafo internamente."""
        g = self.graph  # Richiama implicitamente _ensure_graph()
        return {
            "n_nodes": len(g.nodes()),
            "n_edges": len(g.edges()),
            "nodes_sample": list(g.nodes())[:5]
        }

# Esempio di utilizzo
if __name__ == "__main__":
    
    prot = Protein(pdb_code="3pd9")
    print(f"Informazioni proteina: {prot.get_info()}")
    print(f"Il grafo ha {len(prot.graph.nodes())} nodi.")