"""
Schemas for the features of nodes and edges in the protein-ligand graph.
Each schema is a list of dictionaries, where each dictionary describes a feature.
"""

# ---------------------------------------------------------------------------
# Features Schema -> a dictionary for every feature, column order
# ---------------------------------------------------------------------------

PROTEIN_NODE_SCHEMA = [
    {'type': 'continuous'},                           # meiler 0 — hydrophobicity
    {'type': 'continuous'},                           # meiler 1 — steric volume
    {'type': 'continuous'},                           # meiler 2 — polarizability
    {'type': 'continuous'},                           # meiler 3 — polarity
    {'type': 'continuous'},                           # meiler 4 — isoelectric point
    {'type': 'continuous'},                           # meiler 5 — refractive index
    {'type': 'continuous'},                           # meiler 6 — α-helix propensity
    {'type': 'continuous'},                           # b_factor
    {'type': 'categorical', 'num_categories': 21}     # residue type (20 standard + 1 for non-standard residues)
]

LIGAND_NODE_SCHEMA = [
    {'type': 'categorical', 'num_categories': 120},   # atomic number
    {'type': 'continuous'},                           # degree
    {'type': 'continuous'},                           # total Hs  TODO: continuous or categorical (0/1/2/3+)?
    {'type': 'continuous'},                           # formal charge
    {'type': 'categorical', 'num_categories': 2},     # aromatic (0/1)
    {'type': 'categorical', 'num_categories': 2},     # in ring (0/1)
    {'type': 'categorical', 'num_categories': 7},     # hybridization
]

PROTEIN_EDGE_SCHEMA = [
    {'type': 'categorical', 'num_categories': 2},   # bond type (0=spatial contact, 1=peptide bond)
    {'type': 'continuous'},                         # Cα–Cα distance in Å
]

LIGAND_EDGE_SCHEMA = [
    {'type': 'categorical' , 'num_categories': 5},  # bond type (1=single, 2=double, 3=triple, 4=aromatic, 0=virtual/distance)
    {'type': 'categorical', 'num_categories':2 },   # conjugated (0/1)
    {'type': 'continuous'},                         # 3D distance in Å
]

DATASET_SCHEMAS = {
    'protein_node': PROTEIN_NODE_SCHEMA,
    'protein_edge': PROTEIN_EDGE_SCHEMA,
    'ligand_node': LIGAND_NODE_SCHEMA,
    'ligand_edge': LIGAND_EDGE_SCHEMA
}
