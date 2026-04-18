import networkx as nx
import py3Dmol
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit.Chem import AllChem

# Spostiamo la configurazione fuori per una facile manutenzione
ATOM_PROPERTIES = {
    'H':  ('#FFFFFF', 40),   'F':  ('#CCEEFF', 170), 
    'O':  ('#FF4444', 180),  'N':  ('#4444FF', 190),  
    'C':  ('#AAAAAA', 200),  'Cl': ('#00EE00', 280), 
    'S':  ('#EEEE00', 290),  'P':  ('#FFA500', 300)   
}
DEFAULT_ATOM = ('#00FFCC', 150)

class Ligand:
    def __init__(self, smile=None, sdf_path=None):
        self.mol = self._load_molecule(smile, sdf_path)
        self._computed_3d = False

    def _load_molecule(self, smile, sdf_path):
        """Metodo privato per il caricamento sicuro."""
        mol = None
        if sdf_path:
            supplier = Chem.SDMolSupplier(sdf_path, sanitize=True)
            if supplier: mol = supplier[0]
        elif smile:
            mol = Chem.MolFromSmiles(smile)
        
        if mol is None:
            raise ValueError(f"Errore caricamento: SMILES={smile}, SDF={sdf_path}")
        
        mol = Chem.AddHs(mol)
        return mol

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
    
    def get_features(self):
        """Estrae caratteristiche atomiche e di legame per il grafo."""
        # implementare l'estrazione di feature specifiche per nodi e archi
        pass

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

    def visualize_nx_3d(self):
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

if __name__ == "__main__":
    # Test con Caffeina (più interessante dell'etanolo per i grafi)
    caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    ligand = Ligand(smile=caffeine_smiles)
    ligand.visualize_py3Dmol()
    ligand.visualize_nx_3d()