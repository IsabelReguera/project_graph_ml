import os
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data

from gmnbind.data.protein import Protein
from gmnbind.data.ligand import Ligand


class PDBbindDataset(Dataset):
    """
    On-disk PyG dataset that builds protein OR ligand graphs (selected by `mode`)
    for every complex in the index and saves them as {mode}_{pdb_code}.pt in `output_dir`.

    Complexes whose .pt already exists are skipped, so an interrupted run can be resumed.
    Failures are collected in `self.failures` (status "excluded"); ligands built from SMILES
    because the SDF could not be read are recorded too (status "fallback").
    """

    def __init__(self, root, csv_path, mode, output_dir, structures_dir,
                 from_local_files=True, fallback_to_smiles=True, limit=None,
                 transform=None, pre_transform=None):
        if mode not in ['protein', 'ligand']:
            raise ValueError("'mode' should be 'protein' or 'ligand'")

        self.mode = mode
        self.csv_path = csv_path
        self._output_dir = output_dir
        self.structures_dir = structures_dir
        self.from_local_files = from_local_files
        self.fallback_to_smiles = fallback_to_smiles

        self.failures = []
        self.df = pd.read_csv(csv_path)
        if limit is not None:
            self.df = self.df.iloc[:limit].reset_index(drop=True)

        super().__init__(root, transform, pre_transform)

        # keep only the complexes whose .pt was generated successfully
        valid_indices = []
        for idx, row in self.df.iterrows():
            my_path = os.path.join(self.processed_dir, f'{self.mode}_{row["pdb_code"]}.pt')
            if os.path.exists(my_path):
                valid_indices.append(idx)

        self.df = self.df.loc[valid_indices].reset_index(drop=True)

    @property
    def processed_dir(self):
        return str(self._output_dir)

    @property
    def raw_file_names(self):
        return ['pdbbind_v2020']

    @property
    def processed_file_names(self):
        return [f'{self.mode}_{code}.pt' for code in self.df['pdb_code']]

    def _structure_path(self, pdb_code, suffix):
        return os.path.join(self.structures_dir, pdb_code, f'{pdb_code}_{suffix}')

    def process(self):
        for i, row in self.df.iterrows():
            pdb_code   = row['pdb_code']
            smile      = row['SMILE']
            p_affinity = row['pAffinity']

            save_path = os.path.join(self.processed_dir, f'{self.mode}_{pdb_code}.pt')

            if os.path.exists(save_path):
                continue

            try:
                # --- STEP 1: build the Ligand / Protein object ---
                if self.mode == 'ligand':
                    if self.from_local_files:
                        ligand_path = self._structure_path(pdb_code, 'ligand.sdf')
                        try:
                            obj = Ligand.from_sdf(ligand_path)
                        except Exception as sdf_err:
                            if not self.fallback_to_smiles:
                                raise ValueError(f"SDF not parsable: {sdf_err}")
                            print(f"{i:05d} - invalid SDF for {pdb_code}, falling back to SMILES...")
                            obj = Ligand.from_smile(smile)
                            self.failures.append({"pdb_code": pdb_code, "mode": self.mode,
                                                  "status": "fallback",
                                                  "reason": f"SDF failed ({sdf_err}), used SMILES"})
                    else:
                        obj = Ligand.from_smile(smile)

                elif self.mode == 'protein':  # TODO pocket
                    if self.from_local_files:
                        protein_path = self._structure_path(pdb_code, 'protein.pdb')
                        try:
                            obj = Protein.from_pdb_path(protein_path)
                        except Exception:
                            print(f"{i:05d} - protein {pdb_code} not found locally, downloading from RCSB...")
                            try:
                                obj = Protein.from_pdb_code(pdb_code)
                            except Exception as web_err:
                                raise ValueError(f"Download from RCSB failed too: {web_err}")
                    else:
                        obj = Protein.from_pdb_code(pdb_code)

                # --- STEP 2: extract graph tensors and save ---
                x, edge_index, edge_attr, pos = obj.get_pyg_graph()

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    pos=pos,
                    y=torch.tensor([p_affinity], dtype=torch.float),
                    pdb_code=pdb_code
                )

                torch.save(data, save_path)
                print(f"{i:05d} - {self.mode}_{pdb_code} saved")

            except Exception as e:
                print(f"ERROR processing {self.mode} {pdb_code}: {e}")
                self.failures.append({"pdb_code": pdb_code, "mode": self.mode,
                                      "status": "excluded", "reason": str(e)})
                continue

    def len(self):
        return len(self.df)

    def get(self, idx):
        pdb_code = self.df.iloc[idx]['pdb_code']
        return torch.load(os.path.join(self.processed_dir, f'{self.mode}_{pdb_code}.pt'), weights_only=False)
