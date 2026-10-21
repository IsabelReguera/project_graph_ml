# Import libraries - Settings
#===========================
import os
import re #regex library
import csv
import pandas as pd
import numpy as np
import networkx as nx
from functools import partial

# Graph/bio libraries
import py3Dmol
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plotly_protein_structure_graph
from graphein.protein.edges.distance import add_distance_threshold
from rdkit import Chem

"""#ignore warinings
import warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")"""


#Extract and parse PDBBind index file
#============================
datapath = "data"
filename = "INDEX_general_PL.2020R1.lst"
filepath = os.path.join(datapath, "raw", "pdbbind_v2020", filename)

if not os.path.exists(filepath):
    raise FileNotFoundError(f"{filename} file not found path[{filepath}]")

#read file
with open(filepath, "r", encoding="utf-8") as f:
    lines = [
        ln.strip()
        for ln in f
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    
#regex [PDB_code, resolution, release_year, binding_data[type,compare,unit], reference, ligand_name, comments]
regex= re.compile(r"(\w+)\s+"+                              #pdb code
                  r"([\d.]+|NMR)\s+"+                       #resolution
                  r"(\d{4})\s+"+                            #release year
                  r"(\w+)([=<~]{0,2})([\d\.]+)(\w+)\s+"     #binding data
                  r"//\s[\w\.]+\s+"                         #file path reference (ignored)
                  r"\(([^)]+?)\)"                           #ligand name
                  r"\s*(.*)"                                #comments
                  )

parsed_index = []
skipped = 0
for ln in lines:
    match = regex.match(ln)
    if match:
        (pdb, res, year, btype, comp, val, unit, ligand, comment) = match.groups()
        
        parsed_index.append({
            "pdb_code": pdb,
            "resolution": res,
            "release_year": year,
            "binding_type": btype,
            "compare": comp,
            "value": val,
            "unit": unit,
            "ligand_name": ligand,
            "comments": comment.strip()
        })
    else:
        skipped += 1
        #print("SKIPPED:", ln)
print(f"Parsed: {len(parsed_index)} | Skipped: {skipped}")

#save csv
csv_filename = "index.csv"
csv_filepath = os.path.join(datapath, csv_filename)
print(f"Saving parsed index to {csv_filepath}")
       
