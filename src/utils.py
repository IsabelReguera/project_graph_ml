# ---------------------------------------------------------------------------
# Schemi feature -> un dizionario per ogni feature, nell'ordine delle colonne
# ---------------------------------------------------------------------------

PROTEIN_NODE_SCHEMA = [
    {'type': 'continuous'},                           # meiler 0 — idrofobicità
    {'type': 'continuous'},                           # meiler 1 — volume sterico
    {'type': 'continuous'},                           # meiler 2 — polarizzabilità
    {'type': 'continuous'},                           # meiler 3 — polarità
    {'type': 'continuous'},                           # meiler 4 — pI
    {'type': 'continuous'},                           # meiler 5 — indice di rifrazione
    {'type': 'continuous'},                           # meiler 6 — propensione α-elica
    {'type': 'continuous'},                           # b_factor
    {'type': 'categorical', 'num_categories': 21}     # tipo aminoacido (20 standard + 1 per residui non standard/virtuali)
]

LIGAND_NODE_SCHEMA = [
    {'type': 'categorical', 'num_categories': 120},   # numero atomico
    {'type': 'continuous'},                           # degree
    {'type': 'continuous'},                           # H totali TODO: valutare se lasciare come continua o trasformare in categorica (0/1/2/3+)
    {'type': 'continuous'},                           # carica formale
    {'type': 'categorical', 'num_categories': 2},     # aromaticità (0=falso, 1=vero)
    {'type': 'categorical', 'num_categories': 2},     # in-ring (0=falso, 1=vero)
    {'type': 'categorical', 'num_categories': 7},     # ibridazione
]

PROTEIN_EDGE_SCHEMA = [
    {'type': 'categorical', 'num_categories': 2},   # tipo legame (0=contatto spaziale, 1=peptide bond)
    {'type': 'continuous'},                         # distanza Cα–Cα in Å
]

LIGAND_EDGE_SCHEMA = [
    {'type': 'categorical' , 'num_categories': 5},  # tipo legame (0/1/2/3/4) (1=singolo, 2=doppio, 3=triplo, 4=aromatico, 0=falso/distanza)
    {'type': 'categorical', 'num_categories':2 },   # coniugazione (0=falso, 1=vero)
    {'type': 'continuous'},                         # distanza 3D in Å
]

DATASET_SCHEMAS = {
    'protein_node': PROTEIN_NODE_SCHEMA,
    'protein_edge': PROTEIN_EDGE_SCHEMA,
    'ligand_node': LIGAND_NODE_SCHEMA,
    'ligand_edge': LIGAND_EDGE_SCHEMA
}




