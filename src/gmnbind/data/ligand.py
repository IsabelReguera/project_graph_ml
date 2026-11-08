import networkx as nx
import py3Dmol
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
import torch
import yaml
from pathlib import Path

# default featurization config (used when no cfg is passed, e.g. in the notebooks)
FEATURES_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "data" / "complex_features.yaml"


def load_ligand_config(path=FEATURES_CONFIG):
    """`ligand` block of the featurization config."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["ligand"]


# Silence RDKit logs (errors, warnings, info)
from rdkit import RDLogger
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)


# Atom colors and sizes for the visualization functions
ATOM_PROPERTIES = {
    'H':  ('#FFFFFF', 40),   'F':  ('#CCEEFF', 170), 
    'O':  ('#FF4444', 180),  'N':  ('#4444FF', 190),  
    'C':  ('#AAAAAA', 200),  'Cl': ('#00EE00', 280), 
    'S':  ('#EEEE00', 290),  'P':  ('#FFA500', 300)   
}
DEFAULT_ATOM = ('#00FFCC', 150)

HYBRIDIZATION = {
    rdchem.HybridizationType.S:     0,
    rdchem.HybridizationType.SP:    1,
    rdchem.HybridizationType.SP2:   2,
    rdchem.HybridizationType.SP3:   3,
    rdchem.HybridizationType.SP3D:  4,
    rdchem.HybridizationType.SP3D2: 5,
    rdchem.HybridizationType.OTHER: 6,
}

class Ligand:
    """
    Atom-level ligand graph built with RDKit, from SDF (crystal pose) or SMILES (3D conformer generated).

    cfg: `ligand` block of the featurization config. If None, read from configs/data/complex_features.yaml.
    """
    def __init__(self, mol, is_3d, cfg=None):
        self.mol = mol
        self._computed_3d = is_3d
        self.cfg = cfg if cfg is not None else load_ligand_config()
        
    @classmethod
    def from_smile(cls, smiles, cfg=None):
        """Create a Ligand from a SMILES string (3D coordinates generated lazily)."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol: raise ValueError("Invalid SMILES")
        cfg = cfg if cfg is not None else load_ligand_config()
        if cfg["add_hydrogens"]:
            mol = Chem.AddHs(mol)
        return cls(mol, is_3d=False, cfg=cfg)

    @classmethod
    def from_sdf(cls, path, cfg=None):
        """Create a Ligand from an SDF file (uses the 3D coordinates in the file)."""
        supplier = Chem.SDMolSupplier(path)
        if not supplier: raise ValueError("SDF not found")
        mol = supplier[0]
        if mol is None: raise ValueError(f"Molecule not parsable: {path}")
        cfg = cfg if cfg is not None else load_ligand_config()
        if cfg["add_hydrogens"]:
            mol = Chem.AddHs(mol, addCoords=True)
        return cls(mol, is_3d=True, cfg=cfg)


    def _ensure_3d_coordinates(self):
        """Generate 3D coordinates only if needed (ETKDGv3 + MMFF optimization)."""
        if not self._computed_3d:
            params = AllChem.ETKDGv3()
            res = AllChem.EmbedMolecule(self.mol, params)
            
            if res == -1:  # standard embedding failed: retry with random coordinates
                # useRandomCoords must be set on the params object, not passed as a kwarg
                params_random = AllChem.ETKDGv3()
                params_random.useRandomCoords = True
                AllChem.EmbedMolecule(self.mol, params_random)
                
            AllChem.MMFFOptimizeMolecule(self.mol)
            self._computed_3d = True

    def _get_node_features(self):
        """Atom features, shape [N, 7] (column order must match LIGAND_NODE_SCHEMA):
            atomic number  -> chemical element
            degree         -> number of bonded heavy atoms
            total Hs       -> total number of hydrogens
            formal charge  -> formal charge of the atom
            aromatic       -> 1 if the atom is aromatic
            in ring        -> 1 if the atom is in a ring
            hybridization  -> S, SP, SP2, SP3, SP3D, SP3D2, OTHER (0-6)
        """
        feats = []
        for atom in self.mol.GetAtoms():
            feats.append([
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetTotalNumHs(),
                atom.GetFormalCharge(),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                HYBRIDIZATION.get(atom.GetHybridization(), 6),
            ])
        return torch.tensor(feats, dtype=torch.float32)

    def _get_edges_and_features(self, pos, cutoff=5.0):
        """
        Edges and edge features, including both real chemical bonds and
        "virtual" edges between atoms closer than `cutoff` Å.

        Edge features, shape [E, 3] (must match LIGAND_EDGE_SCHEMA):
            0: bond type (1=single, 2=double, 3=triple, 4=aromatic, 0=virtual/distance)
            1: conjugated (1 if conjugated, 0 otherwise)
            2: 3D distance between the two atoms in Å
        """
        # real chemical bonds
        real_bonds = {}
        for bond in self.mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            real_bonds[(u, v)] = bond
            real_bonds[(v, u)] = bond  # both directions
            
        src, dst, edge_attrs = [], [], []
        num_atoms = self.mol.GetNumAtoms()
        
        # all pairwise 3D distances
        dist_matrix = torch.cdist(pos, pos)
        
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j: 
                    continue  # no self-loops
                
                dist = dist_matrix[i, j].item()
                is_real_bond = (i, j) in real_bonds
                
                # edge if it is a real bond OR the atoms are spatially close
                if is_real_bond or dist <= cutoff:
                    src.append(i)
                    dst.append(j)
                    
                    if is_real_bond:
                        bond = real_bonds[(i, j)]
                        bond_type = bond.GetBondTypeAsDouble()
                        if bond_type == 1.5:  # RDKit uses 1.5 for aromatic: map to 4 to get integer categories
                            bond_type = 4.0
                        is_conjugated = float(bond.GetIsConjugated())
                        
                    else:
                        bond_type = 0.0     # no chemical bond
                        is_conjugated = 0.0  # cannot be conjugated
                        
                        
                    edge_attrs.append([bond_type, is_conjugated, dist])
                    
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        
        return edge_index, edge_attr
   
    def _get_3d_coordinates(self):
        """3D coordinates of the atoms, shape [N, 3]."""
        self._ensure_3d_coordinates()
        conf = self.mol.GetConformer()
        pos = []
        for atom in self.mol.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            pos.append([p.x, p.y, p.z])
        return torch.tensor(pos, dtype=torch.float32)

    def get_pyg_graph(self, distance_cutoff=None):
        """Return (x, edge_index, edge_attr, pos) ready for PyTorch Geometric.
        distance_cutoff: Å for the virtual edges; if None, taken from the config."""
        if distance_cutoff is None:
            distance_cutoff = self.cfg["distance_cutoff"]
        x   = self._get_node_features()
        pos = self._get_3d_coordinates()  
        # edge indices and features computed together
        edge_index, edge_attr = self._get_edges_and_features(pos, cutoff=distance_cutoff)
        return x, edge_index, edge_attr, pos
    
    def to_smile(self):
        """Canonical isomeric SMILES of the ligand."""
        return Chem.MolToSmiles(
            self.mol, 
            isomericSmiles=True,  # keep @ for chiral centers and / \ for double bonds
            kekuleSmiles=False,   # keep aromatic notation (e.g. 'c1ccccc1')
            canonical=True        # unique SMILES for the same structure
            )

    def mol_to_nx(self):
        """NetworkX graph of the chemical bonds with 3D coordinates (for visualization only)."""
        self._ensure_3d_coordinates()
        G = nx.Graph()
        conf = self.mol.GetConformer()

        for atom in self.mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            G.add_node(idx,
                       element=atom.GetSymbol(),
                       x=pos.x, y=pos.y, z=pos.z)

        for bond in self.mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), 
                       bond.GetEndAtomIdx(), 
                       weight=bond.GetBondTypeAsDouble())
        return G

    def visualize_nx3D(self):
        """Static 3D plot with Matplotlib."""
        G = self.mol_to_nx()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # atoms
        for _, d in G.nodes(data=True):
            color, size = ATOM_PROPERTIES.get(d['element'], DEFAULT_ATOM)
            ax.scatter(d['x'], d['y'], d['z'], c=color, s=size, edgecolors='k', alpha=0.85)
            if d['element'] != 'H':
                ax.text(d['x'], d['y'], d['z'], d['element'], 
                        fontsize=8, ha='center', va='center', color='black', weight='bold')

        # bonds
        for u, v, d in G.edges(data=True):
            pos_u, pos_v = G.nodes[u], G.nodes[v]
            ax.plot([pos_u['x'], pos_v['x']], 
                    [pos_u['y'], pos_v['y']], 
                    [pos_u['z'], pos_v['z']], color='gray', lw=d['weight']*2, alpha=0.5)
            
        ax.set_aspect('equal')
        ax.set_axis_off()
        plt.show()

    def visualize_py3Dmol(self):
        """Interactive 3D view with py3Dmol (Jupyter only)."""
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and ip.__class__.__name__ != 'TerminalInteractiveShell':
            self._ensure_3d_coordinates()
            mblock = Chem.MolToMolBlock(self.mol)
            viewer = py3Dmol.view(width=600, height=400)
            viewer.addModel(mblock, 'mol')
            viewer.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
            viewer.zoomTo()
            return viewer.show() 
        else:
            print("py3Dmol requires a Jupyter environment")
            return
    def plot_2d(self):
        """2D drawing with RDKit."""
        from rdkit.Chem import Draw
        img = Draw.MolToImage(self.mol, size=(300, 300))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    ligand = Ligand.from_smile(caffeine_smiles)
    x, edge_index, edge_attr, pos = ligand.get_pyg_graph()
    print(f"Nodes: {x.shape}  |  Edges: {edge_index.shape}  |  Edge attr: {edge_attr.shape}  |  Pos: {pos.shape}")
    ligand.visualize_nx3D()
    
