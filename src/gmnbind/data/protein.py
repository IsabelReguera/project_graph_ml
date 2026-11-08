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

# node annotation functions
node_funcs = [
    meiler_embedding,      # Meiler embedding: 7 physico-chemical properties per amino acid
]

# edge construction functions
edge_funcs = [
    add_peptide_bonds,    # covalent backbone edges
    partial(add_distance_threshold,  long_interaction_threshold=0, threshold=8.0)  # spatial edges (Cα–Cα within 8 Å)
]

# graph configuration with the functions above
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
    """Residue-level protein graph built with Graphein (one node per residue, Cα coordinates)."""
    def __init__(self, pdb_code, pdb_path=None):
        self.pdb_code = pdb_code
        self.pdb_path = pdb_path
        self._graph = None

    @classmethod
    def from_pdb_path(cls, path):
        """Create a Protein from a local PDB file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        else:
            pdb_code = os.path.basename(path).split('_')[0]  # PDB code from the file name
        return cls(pdb_code=pdb_code, pdb_path=path)

    @classmethod
    def from_pdb_code(cls, pdb_code):
        """Create a Protein that downloads the structure from RCSB."""
        if not pdb_code or len(pdb_code) != 4:
            raise ValueError(f"Invalid PDB code: {pdb_code}")
        return cls(pdb_code=pdb_code)

    def _ensure_graph(self):
        """Build the graph lazily, on first access."""
        if self._graph is not None:
            return
        
        config = GRAPH_CONFIG

        if self.pdb_path:
            # Workaround: construct_graph(config=config, path=self.pdb_path) crashes;
            # read the file as a string and parse it with cpdb instead.
            with open(self.pdb_path, "r") as f:
                raw_df = cpdb.parse(pdb_str=f.read())
            self._graph = construct_graph(config=config, df=raw_df)

        else:
            self._graph = construct_graph(config=config, pdb_code=self.pdb_code)

    @property
    def graph(self):
        """NetworkX graph of the protein (built on first access)."""
        self._ensure_graph()
        return self._graph

    def _get_node_features(self):  # TODO: add secondary-structure features (e.g. beta sheet)
        """Residue features, shape [N, 9] (column order must match PROTEIN_NODE_SCHEMA):
            Meiler embedding (7) -> physico-chemical description of the amino acid (hydrophobicity, steric volume,
                                    polarizability, polarity, isoelectric point, refractive index, alpha-helix propensity)
            b_factor             -> flexibility/mobility of the residue (uncertainty of the atom positions)
            residue type         -> integer code of the amino acid (0-19, 20 = unknown), embedded as categorical
        """
        feats = []
        for _, d in self.graph.nodes(data=True):
            meiler = list(d["meiler"].values)       # 7 values
            b_factor = d["b_factor"]                # 1 value
            res_name = AMMINO_ACIDS_DICT.get(d["residue_name"], 20)  # amino acid code (20 = unknown)
            feats.append(meiler + [b_factor] + [res_name])  # concatenate features
        return torch.tensor(feats, dtype=torch.float32)

    def _get_edges_and_features(self):
        """
        Edge features (both directions), shape [2E, 2] (must match PROTEIN_EDGE_SCHEMA):
            0: bond type (0.0 = spatial contact, 1.0 = covalent peptide bond)
            1: Cα–Cα distance in Å
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
        """Cα coordinates of the residues, shape [N, 3] (stored as pos, currently not used by the models)."""
        pos = [d["coords"].tolist() for _, d in self.graph.nodes(data=True)]
        return torch.tensor(pos, dtype=torch.float32)

    def get_pyg_graph(self):
        """Return (x, edge_index, edge_attr, pos) ready for PyTorch Geometric."""
        x                    = self._get_node_features()
        pos                  = self._get_3d_coordinates()
        edge_index, edge_attr = self._get_edges_and_features()
        return x, edge_index, edge_attr, pos
    
    def visualize_protein_graph(self):
        """Plot the protein graph in 3D with Plotly."""
        g = self.graph
    
        fig = plotly_protein_structure_graph(
            g,
            colour_edges_by="kind",
            colour_nodes_by="residue_name",  # color by amino acid type
            node_size_multiplier=0.4,       # node size
            node_alpha=0.7,               # node transparency
            plot_title=f"Graph: {self.pdb_code}"
        )

        # line style for the different interaction types
        fig.update_traces(line=dict(width=3), selector=dict(type='scatter3d'))

        # layout styling
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'  # keep the real proportions of the protein
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig.show()
    
    def visualize_protein_3D(self):
        """Show the protein structure (cartoon) with py3Dmol (Jupyter only)."""
        view = py3Dmol.view(width=800, height=600)

        if self.pdb_path and os.path.exists(self.pdb_path):
            # load from file
            with open(self.pdb_path, 'r') as f:
                pdb_data = f.read()
            view.addModel(pdb_data, 'pdb')
            print(f"Showing local file: {self.pdb_path}")
        elif self.pdb_code:
            # Download
            view = py3Dmol.view(query=f'pdb:{self.pdb_code}')
            print(f"Showing PDB code: {self.pdb_code}")
        else:
            raise ValueError("No PDB code or file path available for visualization.")

        # style
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        # TODO: optionally show side chains of the residues close to the ligand
        view.addStyle({'resn': 'LIG'}, {'stick': {}}) 
        
        view.zoomTo()
        return view.show()


if __name__ == "__main__":
    prot = Protein.from_pdb_code("3pd9")
    x, edge_index, edge_attr, pos = prot.get_pyg_graph()
    print(f"Nodes: {x.shape}  |  Edges: {edge_index.shape}  |  Edge attr: {edge_attr.shape}  |  Pos: {pos.shape}")