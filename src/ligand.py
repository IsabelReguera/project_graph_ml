import networkx as nx
import py3Dmol
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
import torch



# Disabilita i log di RDKit (Errori, Warning, Info)
from rdkit import RDLogger
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)


# Spostiamo la configurazione fuori per una facile manutenzione
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
    """Classe per la gestione dei ligandi, con supporto per SMILES e SDF, e visualizzazione 3D/2D."""
    def __init__(self, mol, is_3d):
        self.mol = mol
        self._computed_3d = is_3d
        
    @classmethod
    def from_smile(cls, smiles):
        """Crea un'istanza di Ligand a partire da una stringa SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol: raise ValueError("SMILES non valido")
        #mol = Chem.AddHs(mol)
        return cls(mol, is_3d=False)

    @classmethod
    def from_sdf(cls, path):
        """Crea un'istanza di Ligand a partire da un file SDF."""
        supplier = Chem.SDMolSupplier(path)
        if not supplier: raise ValueError("SDF non trovato")
        mol = supplier[0]
        if mol is None: raise ValueError(f"Molecola non parsabile: {path}")
        #mol = Chem.AddHs(mol, addCoords=True)
        return cls(mol, is_3d=True)


    def _ensure_3d_coordinates(self):
        """Genera coordinate 3D solo se necessario (Lazy Evaluation)."""
        if not self._computed_3d:
            # Parametri migliorati per l'embedding
            params = AllChem.ETKDGv3()
            res = AllChem.EmbedMolecule(self.mol, params)
            if res == -1: # Se fallisce l'embedding standard, prova il random
                AllChem.EmbedMolecule(self.mol, AllChem.ETKDGv3(), useRandomCoords=True)
                
            AllChem.MMFFOptimizeMolecule(self.mol)
            self._computed_3d = True

    def _get_node_features(self):
        """Feature atomiche: 
            numero atomico -> indicazione dell'elemento chimico
            grado -> numero di legami con altri atomi (esclusi gli idrogeni)
            H totali -> numero totale di idrogeni
            carica formale -> carica elettrica dell'atomo
            aromaticità -> indica se l'atomo è aromatico
            anello -> indica se l'atomo fa parte di un anello
            ibridazione -> tipo di ibridazione dell'atomo (S, SP, SP2, SP3, SP3D, SP3D2, OTHER)
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

    def _get_edge_index(self):
        """Costruisce l'edge_index bidirezionale per PyTorch Geometric."""
        src, dst = [], []
        for bond in self.mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.extend([start, end])
            dst.extend([end, start])
        return torch.tensor([src, dst], dtype=torch.long)

    def _get_edge_features(self):
        """
        Feature di legame (bidirezionali): 
            tipo di legame (singolo=1, doppio=2, triplo=3, aromatico=1.5),
            coniugazione 
        """
        feats = []
        for bond in self.mol.GetBonds():
            f = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsConjugated()),
            ]
            feats.extend([f, f])  # bidirezionale: stesso valore in entrambe le direzioni
        return torch.tensor(feats, dtype=torch.float32)

    def _get_3d_coordinates(self):
        """Estrae le coordinate 3D degli atomi."""
        self._ensure_3d_coordinates()
        conf = self.mol.GetConformer()
        pos = []
        for atom in self.mol.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            pos.append([p.x, p.y, p.z])
        return torch.tensor(pos, dtype=torch.float32)

    def get_pyg_graph(self):
        """Restituisce (x, edge_index, edge_attr, pos) pronti per PyTorch Geometric."""
        x          = self._get_node_features()
        edge_index = self._get_edge_index()
        edge_attr  = self._get_edge_features()
        pos        = self._get_3d_coordinates()
        return x, edge_index, edge_attr, pos
    
    def to_smile(self):
        """Restituisce la rappresentazione SMILES del ligando."""
        return Chem.MolToSmiles(
            self.mol, 
            isomericSmiles=True,  # Include @ per centri chirali e / \ per i doppi legami
            kekuleSmiles=False,   # Mantiene la rappresentazione aromatica (es. 'c1ccccc1')
            canonical=True        # Garantisce che lo SMILES sia unico per quella struttura
            )

    def mol_to_nx(self):
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
        """Visualizzazione statica con Matplotlib."""
        G = self.mol_to_nx()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Disegno nodi
        for _, d in G.nodes(data=True):
            color, size = ATOM_PROPERTIES.get(d['element'], DEFAULT_ATOM)
            ax.scatter(d['x'], d['y'], d['z'], c=color, s=size, edgecolors='k', alpha=0.85)
            if d['element'] != 'H':
                ax.text(d['x'], d['y'], d['z'], d['element'], 
                        fontsize=8, ha='center', va='center', color='black', weight='bold')

        # Disegno legami
        for u, v, d in G.edges(data=True):
            pos_u, pos_v = G.nodes[u], G.nodes[v]
            ax.plot([pos_u['x'], pos_v['x']], 
                    [pos_u['y'], pos_v['y']], 
                    [pos_u['z'], pos_v['z']], color='gray', lw=d['weight']*2, alpha=0.5)
            
        ax.set_aspect('equal')
        ax.set_axis_off()
        plt.show()

    def visualize_py3Dmol(self):
        """Visualizzazione interattiva con py3Dmol (funziona solo in Jupyter)."""
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
            print("!py3Dmol richiede un ambiente Jupyter per la visualizzazione!")
            return
    def plot_2d(self):
        """Visualizzazione 2D con RDKit."""
        from rdkit.Chem import Draw
        img = Draw.MolToImage(self.mol, size=(300, 300))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    ligand = Ligand.from_smile(caffeine_smiles)
    x, edge_index, edge_attr, pos = ligand.get_pyg_graph()
    print(f"Nodi: {x.shape}  |  Archi: {edge_index.shape}  |  Edge attr: {edge_attr.shape}  |  Pos: {pos.shape}")
    ligand.visualize_nx3D()
    
