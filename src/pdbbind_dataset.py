import os
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data

from ligand import Ligand
from protein import Protein

class PDBbindDataset(Dataset):
    def __init__(self, root, csv_path, mode, from_local_files=False, limit=None, transform=None, pre_transform=None):
        if mode not in ['protein', 'ligand']:
            raise ValueError("Il parametro 'mode' deve essere 'protein' o 'ligand'")
            
        self.mode = mode
        self.csv_path = csv_path
        self.from_local_files = from_local_files

        self.df = pd.read_csv(csv_path)
        if limit is not None:
            self.df = self.df.iloc[:limit].reset_index(drop=True)
            
        super().__init__(root, transform, pre_transform)
        
        # Filtro di validità post-processamento
        valid_indices = []
        for idx, row in self.df.iterrows():
            my_path = os.path.join(self.processed_dir, f'{self.mode}_{row["pdb_code"]}.pt')
            if os.path.exists(my_path):
                valid_indices.append(idx)
                
        self.df = self.df.loc[valid_indices].reset_index(drop=True)

    @property
    def raw_file_names(self):
        return ['P-L', 'PL_cleaned_index.csv']

    @property
    def processed_file_names(self):
        return [f'{self.mode}_{code}.pt' for code in self.df['pdb_code']]

    def process(self):
        for i, row in self.df.iterrows():
            pdb_code   = row['pdb_code']
            smile      = row['SMILE']
            p_affinity = row['pAffinity']
            year       = row['release_year']
            
            save_path = os.path.join(self.processed_dir, f'{self.mode}_{pdb_code}.pt')
            
            if os.path.exists(save_path):
                continue
                
            try:
                # --- FASE 1: CREAZIONE OGGETTO (Ligand o Protein) ---
                if self.mode == 'ligand':
                    if self.from_local_files:
                        ligand_path = os.path.join(self.raw_dir, 'P-L', year, pdb_code, f'{pdb_code}_ligand.sdf')
                        try:
                            obj = Ligand.from_sdf(ligand_path)
                        except Exception:
                            print(f"{i:05d} - SDF non valido per {pdb_code}, fallback su SMILES...")
                            obj = Ligand.from_smile(smile)
                    else:
                        obj = Ligand.from_smile(smile)
                        
                elif self.mode == 'protein':
                    if self.from_local_files:
                        pocket_path = os.path.join(self.raw_dir, 'P-L', year, pdb_code, f'{pdb_code}_pocket.pdb')
                        try:
                            obj = Protein.from_pdb_path(pocket_path)
                        except Exception:
                            print(f"{i:05d} - Proteina {pdb_code} non trovata localmente. Scaricamento dal web...")
                            try:
                                obj = Protein.from_pdb_code(pdb_code)
                            except Exception as web_err:
                                raise ValueError(f"Fallito anche il download dal web: {web_err}")
                    else:
                        obj = Protein.from_pdb_code(pdb_code)

                # --- FASE 2: ESTRAZIONE E SALVATAGGIO ---
                # Se il codice arriva a questa riga, l'oggetto 'obj' è stato creato con successo
                x, edge_index, edge_attr, pos = obj.get_pyg_graph()

                # Definiamo la variabile 'data' usando la classe Data di PyTorch Geometric
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    pos=pos,
                    y=torch.tensor([p_affinity], dtype=torch.float),
                    pdb_code=pdb_code
                )

                torch.save(data, save_path)
                print(f"{i:05d} - {self.mode}_{pdb_code} salvato con successo!")
                
            except Exception as e:
                # Qualsiasi errore (anche interno alla libreria RDKit o di rete) viene stampato qui
                print(f"ERRORE CRITICO elaborando {self.mode} {pdb_code}: {e}")
                continue

    def len(self):
        return len(self.df)

    def get(self, idx):
        pdb_code = self.df.iloc[idx]['pdb_code']
        return torch.load(os.path.join(self.processed_dir, f'{self.mode}_{pdb_code}.pt'), weights_only=False)


# --- BLOCCO MAIN ---
if __name__ == "__main__":
    
    print("--- INIZIO GENERAZIONE PROTEINE ---")
    protein_dataset = PDBbindDataset(root='data', csv_path='data/raw/PL_cleaned_index.csv', mode='protein', limit=1000, from_local_files=True)
    
    print("\n--- INIZIO GENERAZIONE LIGANDI ---")
    ligand_dataset = PDBbindDataset(root='data', csv_path='data/raw/PL_cleaned_index.csv', mode='ligand', limit=1000, from_local_files=True)
    # Allineamento dei dataset
    valid_proteins = set(protein_dataset.df['pdb_code'])
    valid_ligands = set(ligand_dataset.df['pdb_code'])
    common_pdbs = list(valid_proteins.intersection(valid_ligands))
    
    protein_dataset.df = protein_dataset.df[protein_dataset.df['pdb_code'].isin(common_pdbs)].reset_index(drop=True)
    ligand_dataset.df = ligand_dataset.df[ligand_dataset.df['pdb_code'].isin(common_pdbs)].reset_index(drop=True)

    print(f"\nGenerazione completata! Complessi validi e allineati: {len(protein_dataset)}")
    
    # CONTROLLO DI SICUREZZA: Eseguiamo il test solo se almeno 1 complesso è sopravvissuto
    if len(protein_dataset) > 0:
        prot_sample = protein_dataset[0]
        lig_sample = ligand_dataset[0]
        
        print("\n--- TEST MATCH ---")
        print(f"Match PDB ID: {prot_sample.pdb_code} == {lig_sample.pdb_code}")
        print(f"Match Score (Y): {prot_sample.y.item():.2f} == {lig_sample.y.item():.2f}")
    else:
        print("\nNessun complesso è sopravvissuto all'estrazione. Controllare i log di errore sopra.")