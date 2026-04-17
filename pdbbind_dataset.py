import os
import pandas as pd
import torch
from torch_geometric.data import Dataset, HeteroData
import numpy as np

class PDBbindHeteroDataset(Dataset):
    def __init__(self, root, csv_path, transform=None, pre_transform=None):
        self.csv_path = csv_path
        
        #Caricamento dataset
        self.df = pd.read_csv(csv_path)
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # I file grezzi che ci aspettiamo di trovare in root/raw/
        return ['P-L', 'parsed_index.csv']

    @property
    def processed_file_names(self):
        # I file che verranno salvati alla fine. PyG controlla se esistono 
        # per decidere se saltare o eseguire la funzione process()
        return [f'complex_{i}.pt' for i in range(len(self.df))]

    def process(self):
        """Metodo principale che elabora e salva i grafi."""
        for idx, row in self.df.iterrows():
            pdb_code = row['pdb_code']
            p_affinity = row['pAffinity']
            

            # 2. instanziazione di una proteina e di un ligando a partire dal csv


            # 3. Creazione del grafo tramite Graphein / RDKit (Logica Custom)
            # g_protein = construct_protein_graph(pdb_path=pocket_path, config=...)
            # feature_proteina, edge_proteina, pos_proteina = formatta_output(g_protein)
            # feature_ligando, edge_ligando, pos_ligando = elabora_sdf_con_rdkit(ligand_path)
            # edge_bipartiti = calcola_interazioni_entro_5_angstrom(pos_proteina, pos_ligando)

            # 4. Assegnazione in HeteroData
            data = HeteroData()
            
            # Qui si inseriscono i tensori reali (questa è una simulazione dell'assegnazione)
            # data['protein'].x = feature_proteina
            # data['protein'].pos = pos_proteina
            # data['ligand'].x = feature_ligando
            # data['ligand'].pos = pos_ligando
            
            # Matrici di connettività
            # data['protein', 'covalent', 'protein'].edge_index = edge_proteina
            # data['ligand', 'covalent', 'ligand'].edge_index = edge_ligando
            # data['protein', 'interacts', 'ligand'].edge_index = edge_bipartiti
            
            # Variabile bersaglio
            data.y = torch.tensor([p_affinity], dtype=torch.float)
            data.pdb_id = pdb_code # Opzionale, per tracciare il file

            # 5. Salvataggio del complesso
            # Se hai filtri applicati con self.pre_filter(data), richiamali qui
            save_path = os.path.join(self.processed_dir, f'complex_{idx}.pt')
            torch.save(data, save_path)

    def len(self):
        return len(self.df)

    def get(self, idx):
        """Permette al DataLoader di caricare un grafo in modo asincrono (Lazy Loading)."""
        data = torch.load(os.path.join(self.processed_dir, f'complex_{idx}.pt'))
        return data