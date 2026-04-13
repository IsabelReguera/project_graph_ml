#-----------------------------------------------------------------------------------------------------------
# Esperimenti con RDKit e NetworkX per visualizzazione di molecole come grafi e calcolarne proprietà chimiche
#-----------------------------------------------------------------------------------------------------------

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import networkx as nx
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

SEED = 104

df= pd.read_csv('data/mini_dataset_chembl.csv')
smiles = df['smiles'].tolist()
mols = [Chem.MolFromSmiles(s) for s in smiles]

atom_properties = { # (color, size)
    'H':  ('#FFFFFF', 40),   
    'F':  ('#CCEEFF', 170), 
    'O':  ('#FF4444', 180),  
    'N':  ('#4444FF', 190),  
    'C':  ('#AAAAAA', 200), 
    'Cl': ('#00EE00', 280), 
    'S':  ('#EEEE00', 290), 
    'P':  ('#FFA500', 300)   
}
default_atom = ('#00FFCC', 150)


def mol_to_nx(mol):
    #Aggiungo H impliciti e calcolo coordinate 3D 
    mol_with_H = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_H, randomSeed=SEED)
    AllChem.MMFFOptimizeMolecule(mol_with_H)

    # Estraggo le coordinate
    conformer = mol_with_H.GetConformer()
    G = nx.Graph()
    for atom in mol_with_H.GetAtoms():

        idx=atom.GetIdx()
        symbol=atom.GetSymbol()
        pos=conformer.GetAtomPosition(idx)

        G.add_node(idx,
                   element=symbol, 
                   x=pos.x, 
                   y=pos.y, 
                   z=pos.z)

    for bond in mol_with_H.GetBonds():
        bond_type = bond.GetBondTypeAsDouble()
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond_type)
        
    return G

def visualize_nx_3D(G):
    fig= plt.figure(figsize=(15, 15))
    
    ax= fig.add_subplot(111, projection='3d')
    ax.set_title("Molecule 3D Graph")
    
    for _, data in G.nodes(data=True):
        x, y, z = data['x'], data['y'], data['z']
        element = data['element']
        color, size = atom_properties.get(element, default_atom)
        
        ax.scatter(x, y, z, color=color, s=size, edgecolors='k', alpha=0.9)
        if element != 'H':  # Non mostrare etichette per gli idrogeni
            ax.text(x, y, z, element, fontsize=6, weight='bold', ha='center', va='center', color='black')
    
    for node1, node2, data in G.edges(data=True):
        x1, y1, z1 = G.nodes[node1]['x'], G.nodes[node1]['y'], G.nodes[node1]['z']
        x2, y2, z2 = G.nodes[node2]['x'], G.nodes[node2]['y'], G.nodes[node2]['z']
        bond= data['bond_type']
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='gray', linewidth=bond*2)
        
    ax.set_aspect('equal')
    ax.axis('off')
    
    #Legenda
    mol_elements = set(nx.get_node_attributes(G, 'element').values())
    legend = []
    for elem in mol_elements:
        color, _ = atom_properties.get(elem, default_atom)
        dot = Line2D([0], [0], marker='o', color='w', label=elem, markerfacecolor=color, markeredgecolor='dimgray', markersize=10)
        legend.append(dot)
        
    ax.legend(handles=legend, title="Atomi", loc="upper left", frameon=True)
    
    plt.show()
        

caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
mol = Chem.MolFromSmiles(caffeine_smiles)
G = mol_to_nx(mol)
visualize_nx_3D(G)
  