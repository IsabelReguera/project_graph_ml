import os
import pandas as pd
import torch
from torch_geometric.data import Dataset, HeteroData

from ligand import Ligand
from protein import Protein

class PDBbindHeteroDataset(Dataset):
    def __init__(self, root, csv_path, from_local_files=False, limit=None, transform=None, pre_transform=None):
        self.csv_path = csv_path
        self.from_local_files = from_local_files

        self.df = pd.read_csv(csv_path)
        if limit is not None:
            self.df = self.df.iloc[:limit].reset_index(drop=True)
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # I file grezzi che ci aspettiamo di trovare in root/raw/
        return ['P-L', 'PL_cleaned_index.csv']

    @property
    def processed_file_names(self):
         # I file processati che ci aspettiamo di trovare in root/processed/
        return [f'complex_{code}.pt' for code in self.df['pdb_code']]

    def process(self):
        """Elabora ogni complesso e salva il grafo su disco."""
        for i, row in self.df.iterrows():
            
            pdb_code    = row['pdb_code']
            smile       = row['SMILE']
            p_affinity  = row['pAffinity']
            year        = row['release_year']
            
            # 0. Definiamo il percorso finale del file
            save_path = os.path.join(self.processed_dir, f'complex_{pdb_code}.pt')
            
            if os.path.exists(save_path):
                print(f"{i:05d} - complex_{pdb_code}.pt already exists. Skipping...") 
                continue
            
            else:
                
                try:

                    # 1. Caricamento proteina e ligando
                    if self.from_local_files:
                        pocket_path = os.path.join(self.raw_dir, 'P-L', year, pdb_code, f'{pdb_code}_pocket.pdb')
                        ligand_path = os.path.join(self.raw_dir, 'P-L', year, pdb_code, f'{pdb_code}_ligand.sdf')
                        protein = Protein.from_pdb_path(pocket_path)
                        ligand  = Ligand.from_sdf(ligand_path)
                    else:
                        protein = Protein.from_pdb_code(pdb_code)
                        ligand  = Ligand.from_smile(smile)

                    # 2. Estrazione tensori
                    x_protein, edge_index_protein, edge_attr_protein, pos_protein = protein.get_pyg_graph()
                    x_ligand,  edge_index_ligand,  edge_attr_ligand,  pos_ligand  = ligand.get_pyg_graph()
                    # edge_bipartiti = calcola_interazioni(pos_proteina, pos_ligando) ?? possibile aggiunta


                    # 3. Assegnazione tensori a HeteroData
                    data = HeteroData()

                    data['protein'].x          = x_protein
                    data['protein'].pos        = pos_protein #?? hanno senso solo se lette da file non so se serve
                    data['protein'].edge_index = edge_index_protein
                    data['protein'].edge_attr  = edge_attr_protein
                    #potenzialmente implementabile per distinguere i diversi tipi di legami
                    #data['protein', 'covalent', 'protein'].edge_index = edge_index_covalente
                    #data['protein', 'contact', 'protein'].edge_index = edge_index_spaziale

                    data['ligand'].x           = x_ligand
                    data['ligand'].pos         = pos_ligand  #?? hanno senso solo se lette da file non so se serve
                    data['ligand'].edge_index  = edge_index_ligand
                    data['ligand'].edge_attr   = edge_attr_ligand
                    # data['protein', 'interacts', 'ligand'].edge_index = edge_bipartiti ?? possibile aggiunta ma non so se serve

                    #target
                    data.y = torch.tensor([p_affinity], dtype=torch.float)
                    data.pdb_code = pdb_code

                    # 5. Salvataggio del complesso
                    # filtri applicati con self.pre_filter(data) ?? se vogliamo escludere alcuni complessi pre salvataggio 
                    torch.save(data, os.path.join(self.processed_dir, f'complex_{pdb_code}.pt'))
                    print(f"{i:05d} - complex_{pdb_code} processed and saved")
                    
                except Exception as e:
                    print(f"Error processing {pdb_code}: {e}")
                    continue
            

    def len(self):
        return len(self.df)

    def get(self, idx):
        """Permette al DataLoader di caricare un grafo in modo asincrono (Lazy Loading)."""
        pdb_code = self.df.iloc[idx]['pdb_code']
        return torch.load(os.path.join(self.processed_dir, f'complex_{pdb_code}.pt'), weights_only=False)
    

# Esempio di utilizzo
if __name__ == "__main__":
    dataset = PDBbindHeteroDataset(root='data', csv_path='data/raw/PL_cleaned_index.csv',limit=1000, from_local_files=True)
    print(f"Dataset caricato. Numero di complessi: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample 0 - PDB ID: {sample.pdb_code}, Affinity: {sample.y.item():.2f}")
    print(f"Protein shape: {sample['protein'].x.shape}, Edge index shape: {sample['protein'].edge_index.shape}, Edge attr shape: {sample['protein'].edge_attr.shape}")
    print(f"Ligand shape: {sample['ligand'].x.shape}, Edge index shape: {sample['ligand'].edge_index.shape}, Edge attr shape: {sample['ligand'].edge_attr.shape}")