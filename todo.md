- guarda paper modello cross attention https://github.com/Lin-Yijie/Graph-Matching-Networks
- cerca di capire come fare baseline (modello matching senza cross attention)
- tokenizzazione features -> embeddo tutte le feature in uno spazio uguale all'input della rete
- cercare info su metrica di valutazione -> se fare classificazione o regressione e su quale valore


# Piano: Modello GMN-style per Binding Affinity Proteina–Ligando
Context
Il progetto ha già il preprocessing completo (Protein + Ligand → PyG Data su disco). Il passo successivo è costruire un modello che prenda i due grafi come input e predica pAffinity (valore continuo = -log10(Kd/Ki/IC50)). L'ispirazione è Graph Matching Networks (GMN, Li et al. ICML 2019), adattata: invece di verificare se due grafi sono strutturalmente simili, il modello deve imparare quanto proteina e ligando si "matchano" biologicamente.

Analisi del problema e parere step-by-step
## Step 0 — Usa il pocket, non la proteina intera
Il dataset ha già {pdb_code}_pocket.pdb (residui entro ~10 Å dal ligando). Usare sempre il pocket invece di _protein.pdb:

La proteina intera ha ~300–1000 residui; il pocket ne ha ~30–80
Cross-attention tra pocket e ligando è O(N_pocket × N_ligand) ≈ 50×30 = gestibile
Cross-attention su proteina intera sarebbe O(500×30) = costoso e rumoroso
Impatto su pdbbind_dataset.py: aggiungere mode='pocket' che usa _pocket.pdb invece di _protein.pdb.

## Step 1 — Feature Tokenization per-feature (comune a tutti i modelli)
Ogni feature viene espansa individualmente in uno spazio token_dim — non una proiezione congiunta. Ogni feature diventa un token:
- feature continua  → Linear(1, token_dim)               — no ReLU
- feature categorica → nn.Embedding(num_categories, token_dim)

I token vengono concatenati e aggregati con un MLP:
[tok_0 ∥ tok_1 ∥ ... ∥ tok_N]  →  Linear(N·token_dim, hidden_dim) → ReLU 

Schema feature proteina (9 feature): 7 Meiler (continui) + b_factor (continuo) + aa_code (categorico, 20 classi)
Schema feature ligando  (7 feature): atomic_number (categorico, 120) + degree, total_h, formal_charge, aromatic, in_ring (continui) + hybridization (categorico, 7)
Edge features: proiezione congiunta Linear(edge_in, edge_dim) senza ReLU — sono tutte continue e poche (2 o 4).

Classe: FeatureTokenizer in src/featuresembedding.py

## Step 2 — Baseline (GNN senza cross-attention)
Architettura:

protein_nodes → Tokenizer → GNN × L strati → mean pooling → h_P  [hidden_dim]
ligand_nodes  → Tokenizer  → GNN × L strati → mean pooling → h_L  [hidden_dim]

concat([h_P, h_L])  →  MLP(2·hidden_dim → hidden_dim → 1)  →  pAffinity_pred
Layer GNN consigliato: GINEConv (PyG) perché accetta edge features natively.

Il baseline è il punto di riferimento: se il modello cross-attention non batte questo, la cross-attention non aggiunge valore.

## Step 3 — Modello Cross-Attention GMN-style
Ogni layer GNN viene esteso con un passo di cross-graph attention:

Per ogni nodo proteina i:

a_ij = softmax_j( sim(h_i^P, h_j^L) )        # attenzione su tutti i nodi ligando
cross_msg_i = sum_j( a_ij · h_j^L )           # messaggio cross-grafo
h_i^P_new = MLP( [intra_msg_i ∥ cross_msg_i] )
Simmetricamente per ogni nodo ligando verso la proteina.

sim può essere dot-product diviso per sqrt(d), oppure distanza euclidea negativa (come nel paper GMN).

Dopo L layer con cross-attention:

mean_pool(H^P) → h_P
mean_pool(H^L) → h_L
MLP([h_P ∥ h_L]) → pAffinity
Differenza chiave vs. baseline: durante il message passing i nodi dei due grafi si "parlano" → il modello impara quali residui interagiscono con quali atomi.

## Step 4 — Task: Regressione su pAffinity
Usare regressione, non classificazione.

Motivazioni:

pAffinity è un valore continuo (-log10 dell'affinità): perdere questa informazione è controproducente
La classificazione (attivo/inattivo) richiederebbe una soglia arbitraria
Tutti i benchmark standard (CASF-2016) usano regressione
Loss: MSE o Huber loss (più robusta a outlier)

Metriche di valutazione (standard nel campo):

RMSE (root mean squared error) — metrica primaria
Pearson R — correlazione lineare
Spearman ρ — correlazione di rango (robusta a outlier)
Concordance Index (CI) — opzionale, misura quanto l'ordine relativo è preservato

## Step 5 — DataLoader che accoppia i due grafi
pdbbind_dataset.py produce protein_{code}.pt e ligand_{code}.pt separatamente. Serve un collator che li unisca per pdb_code in batch:

class ComplexDataset(Dataset):
    def __getitem__(self, idx):
        protein_data = torch.load(f"processed/protein_{code}.pt")
        ligand_data  = torch.load(f"processed/ligand_{code}.pt")
        return protein_data, ligand_data   # stessa pAffinity in entrambi

**collate_fn: raggruppa batch di coppie (Batch_protein, Batch_ligand)**

Questo è necessario prima di qualsiasi training.

Ordine di implementazione raccomandato
Cosa    -	 File
1.	Aggiungere mode='pocket' a pdbbind_dataset.py	    src/pdbbind_dataset.py
2.	ComplexDataset + collate_fn per coppie	            src/complex_dataset.py (nuovo)
3.	Feature tokenization (encoder separati, condivisi)	src/encoders.py (nuovo)
4.	Baseline GNN (GINEConv + pooling + MLP)	            src/baseline_model.py (nuovo)
5.	Cross-attention layer + GMN model completo	        src/gmn_model.py (nuovo)
6.	Training loop con MSE + metriche RMSE/R/Spearman	src/train.py (nuovo)
Nota sulla separazione:

#### info: 
- src/encoders.py — encoder di feature condivisi(importati da entrambi i modelli)
- src/baseline_model.py — BaselineGNN: due GNN indipendenti + MLP finale
- src/gmn_model.py — GMNModel: stessi encoder + layer con cross-attention
- Verifica

# Dalla root del repo
source .venv/Scripts/activate
python -m src.complex_dataset   # smoke test: stampa shape di un batch
python -m src.train --model baseline --limit 100   # overfitting su 100 sample → loss scende
python -m src.train --model gmn    --limit 100     # stesso test con cross-attention


# Note di design (non bug, ma da valutare)
Aspetto	Attuale	Alternativa
Q/K/V projectors	Condivisi tra proteina e ligando	Separati per i due domini
CONNESSIONI RESIDUALI	Assenti	h_new = h_old + f(...) per stabilità !!!!!
Normalizzazione	Assente tra i layer	LayerNorm dopo ogni GMNLayer (come nella baseline)
Edge features nei layer successivi	Invariate (solo .x viene aggiornato)	Aggiornamento degli archi a ogni layer
Vuoi che corregga i due bug direttamente nel codice?

## edge features
Ad ogni iterazione, solo prot.x e lig.x (i vettori dei nodi) vengono aggiornati. prot.edge_attr e lig.edge_attr (i vettori degli archi) rimangono sempre quelli prodotti dal tokenizer all'inizio — layer 1, layer 2, layer 3 usano tutti le stesse edge features originali.

Questo è il comportamento standard della maggior parte delle GNN: i nodi si "parlano" e si aggiornano, gli archi restano statici. Nel tuo caso non è un problema — le edge features (distanza, tipo di legame) sono proprietà geometriche fisse della molecola e ha senso non modificarle.

L'alternativa "aggiornamento degli archi" prevederebbe un MLP separato che, a ogni layer, ricalcola edge_attr in funzione dei nodi sorgente e destinazione aggiornati:


# esempio concettuale
new_edge_attr = f(x[src], x[dst], old_edge_attr)
Questo è ciò che fa ad esempio MPNN (Gilmer et al. 2017), ma per il tuo task è probabilmente superfluo.


# ATTENTION COLLASSA -come fixarre
- normalizzare embedding -> fatto -> X cambia poco nulla (anzi peggiora)
- normalizzare attention -> da provare IMPORTANTE
- inserire LR dinamico con warmup -> fatto


# plot della distribuzione affinity
- capire come settare treshold * (potenziale aggiunta paper)