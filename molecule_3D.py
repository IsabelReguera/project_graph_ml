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
import pandas as pd
import py3Dmol
import os
import webbrowser

SEED = 104

df= pd.read_csv('data/mini_dataset_chembl.csv')
smiles = df['smiles'].tolist()
mols = [Chem.MolFromSmiles(s) for s in smiles]

color_map_atoms = {
    'C':  '#AAAAAA', # Grigio 
    'O':  '#FF4444', # Rosso acceso
    'N':  '#4444FF', # Blu acceso
    'S':  '#EEEE00', # Giallo scuro
    'P':  '#FFA500', # Arancione
    'F':  '#CCEEFF', # Celeste pallido
    'Cl': '#00EE00', # Verde brillante
    'H':  '#FFFFFF'  # Bianco
}
default_color_atom = '#FFD700'

def mol_to_3D(mol):
    # Aggiungo idrogeni impliciti per una rappresentazione più completa (necessario per 3D)
    mol_with_H = Chem.AddHs(mol)
    #print(f"Smile con idrogeni espliciti : {Chem.MolToSmiles(mol_with_H)}")

    # Genero coordinate 3D e ottimizzo la geometria della molecola
    AllChem.EmbedMolecule(mol_with_H, randomSeed=SEED)
    AllChem.MMFFOptimizeMolecule(mol_with_H)
    #print(f"Smile con idrogeni espliciti ottimizzato : {Chem.MolToSmiles(mol_with_H)}")

    # Estraiamo le coordinate
    molblock = Chem.MolToMolBlock(mol_with_H)

    # 2. Configurazione del visualizzatore py3Dmol
    viewer = py3Dmol.view(width=800, height=800)
    viewer.addModel(molblock, 'mol')
    viewer.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.3}})
    viewer.zoomTo()

    # Otteniamo il codice HTML generato da py3Dmol
    html_content = viewer._make_html()

    # Salviamo l'HTML in un file temporaneo locale
    html_file = "mol_3d_temp.html"
    with open(html_file, "w") as f:
        f.write(html_content)

    # Otteniamo il percorso assoluto del file per evitare problemi con il browser
    abs_path = f"file://{os.path.abspath(html_file)}"

    # Chiediamo a Python di aprire il file nel tuo browser predefinito (Chrome, Firefox, Safari, ecc.)
    print("3D view opening in the browser...")
    webbrowser.open(abs_path)

def mol_to_nx(mol):
    #Aggiungo H impliciti e calcolo coordinate 3D 
    mol_with_H = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_H, randomSeed=SEED)
    AllChem.MMFFOptimizeMolecule(mol_with_H)

    # Estraggo le coordinate
    conformer = mol_with_H.GetConformer()
    G = nx.Graph()
    for atom in mol.GetAtoms():

        idx=atom.GetIdx()
        symbol=atom.GetSymbol()
        pos=conformer.GetAtomPosition(idx)

        G.add_node(idx,
                   element=symbol, 
                   x=pos.x, 
                   y=pos.y, 
                   z=pos.z)

    for bond in mol.GetBonds():
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
        color = color_map_atoms.get(element, default_color_atom)
        
        ax.scatter(x, y, z, color=color, s=100)
        ax.text(x, y, z, element, fontsize=8, ha='center', va='center')
    
    for node1, node2, data in G.edges(data=True):
        x1, y1, z1 = G.nodes[node1]['x'], G.nodes[node1]['y'], G.nodes[node1]['z']
        x2, y2, z2 = G.nodes[node2]['x'], G.nodes[node2]['y'], G.nodes[node2]['z']
        bond= data['bond_type']
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='gray', linewidth=bond*2)
        
    plt.show()
        

for mol in mols[:1]:  
    mol_to_3D(mol)
    G = mol_to_nx(mol)
    visualize_nx_3D(G)
  