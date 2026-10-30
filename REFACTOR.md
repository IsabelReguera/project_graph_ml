# Refactor della repo — stato e cose da fare

Documento di passaggio di consegne per il refactor del branch `refactor/cleanup`.
Ultimo aggiornamento: 29/10/2026 (Andrea). Ultimo commit al momento della scrittura: `9a1ef37`.

---

## 0. Principi del refactor (da rispettare)

1. **Il refactor non deve cambiare nessun risultato.** Prima si mette in ordine, poi si correggono i bug su una base pulita. Se si mescolano le due cose e i numeri cambiano, non si sa più perché.
2. **I bug noti NON si correggono durante il refactor**: si annotano (sezione 6) e diventano issue/branch separati dopo il merge.
3. **Un commit per ogni passo**, piccolo e con un messaggio chiaro. Dopo ogni commit verificare almeno:
   ```bash
   python -c "import gmnbind.models.gmn, gmnbind.models.baseline, gmnbind.data.datamodule"
   ```
4. Si lavora sul branch `refactor/cleanup`; alla fine si apre una **PR verso `main`** (review dell'altro, poi squash o merge).
5. Codice, commenti e README in **inglese** (la repo accompagnerà il paper).

---

## 1. Setup dell'ambiente (Isabel)

```bash
git fetch origin
git switch refactor/cleanup

# venv nuovo (quello vecchio puntava alla cartella GML_repo e non funziona più)
python -m venv .venv
source .venv/Scripts/activate          # Git Bash su Windows
python -m pip install --upgrade pip

# GPU: il torch di PyPI su Windows è solo CPU -> installarlo prima dall'indice PyTorch
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # cu128 dipende dalla versione CUDA

pip install -r requirements.txt
pip install -e .                       # installa il pacchetto gmnbind in modalità editable

python -c "import gmnbind; print(gmnbind.__file__)"   # deve puntare a .../src/gmnbind/__init__.py
```

Note:
- `pip install -e .` va fatto **sempre con `-e`**: così le modifiche ai file in `src/` sono subito visibili senza reinstallare.
- Se dopo aver rinominato/spostato la cartella `pip` dà "Fatal error in launcher", il venv è rotto: va ricreato.

**Dati grezzi** (fuori da git): PDBbind v2020 general set in
```
data/raw/pdbbind_v2020/
├── INDEX_general_PL.2020R1.lst
└── generalset/{pdb_code}/{pdb_code}_protein.pdb | _pocket.pdb | _ligand.sdf | _ligand.mol2
```

---

## 2. Stato della repo

### Branch e tag

| Nome | Cosa è |
|---|---|
| tag `LIRE-2026` | stato del codice usato per il paper consegnato (non si sposta mai: `git checkout LIRE-2026` per tornarci) |
| `archive/LIRE-2026` | branch di archivio dello stesso stato |
| `archive/optuna` | vecchio branch con il tuning Optuna (`tune.py`, scaling): da guardare se serve recuperare qualcosa |
| `origin/compa` | vecchio branch remoto, probabilmente da archiviare o cancellare |
| `main` | ancora la versione vecchia (pre-refactor) |
| `refactor/cleanup` | **branch di lavoro**, già pushato su origin |

### Struttura attuale

```
gmn-binding_repo/
├── pyproject.toml                 # pacchetto gmnbind (src layout)
├── requirements.txt
├── README.md                      # da aggiornare (vedi R9)
├── todo.md                        # vecchio, da sostituire con questo file / issue
├── configs/
│   ├── baseline.yaml              # config esperimento (data/model/training)
│   ├── gmn.yaml
│   └── data/
│       ├── index_v2020.yaml           # filtri indice: solo Kd, res <= 2.5 Å (setup del paper LIRE)
│       ├── index_v2020_generalset.yaml# filtri indice: Kd+Ki, res <= 2.75 Å
│       └── complex_features.yaml      # config CORRENTE della featurizzazione (campo `version`)
├── data/
│   ├── raw/                       # fuori da git
│   ├── index/                     # IN git: indici + split
│   │   ├── index.csv              # output di build_index.py (4852 complessi, setup Kd)
│   │   ├── PL_cleaned_index.csv   # vecchio indice (da rimuovere, vedi R1)
│   │   ├── parsed_index.csv       # vecchio parse del .lst (da rimuovere, vedi R1)
│   │   ├── splits/                # vuota (vedi R6)
│   │   └── cv5/                   # vuota (vedi R6)
│   └── processed/
│       ├── v1/                    # .pt fuori da git, meta.json IN git
│       └── test/                  # cartella di prova, da rimuovere (vedi R1)
├── notebooks/                     # solo esplorazione e figure
├── scripts/
│   ├── build_index.py
│   ├── preprocess.py
│   └── train.py
└── src/gmnbind/
    ├── data/
    │   ├── protein.py, ligand.py  # featurizzazione (+ visualizzazione, da separare: R4)
    │   ├── schemas.py             # schemi delle feature (DATASET_SCHEMAS)
    │   ├── pdbbind_dataset.py     # genera i .pt (usato da preprocess.py)
    │   ├── complex_dataset.py     # carica coppie (protein, ligand) + collate
    │   └── datamodule.py          # ComplexDataModule (split casuale 80/20 per ora)
    └── models/
        ├── tokenizer.py           # FeatureTokenizer, DualGraphTokenizer (+ GraphPropLayer legacy)
        ├── layers.py              # CrossGraphAttention, GMNLayer
        ├── base.py                # VUOTO (vedi R3)
        ├── baseline.py
        └── gmn.py
```

---

## 3. Cosa è stato fatto

| Commit | Cosa |
|---|---|
| `5158aad` | Spostamento dei file nella nuova struttura **senza modifiche al codice** (git riconosce le rinomine, la storia dei file è conservata). Riscritto `.gitignore`. |
| `ac7fa21` | `pyproject.toml` + `__init__.py` → `gmnbind` è un pacchetto installabile. `requirements.txt` riscritto (aggiunti `torchmetrics`, `scipy`, `ipykernel`, `nbstripout`). `build_index.py` funzionante. |
| `46bc823` | `scripts/utils.py` smontato in `src/gmnbind/data/schemas.py` + `src/gmnbind/data/datamodule.py`. `train.py` con import del pacchetto, `--config`, percorsi risolti da `REPO_ROOT`. |
| `9a1ef37` | `scripts/preprocess.py` + sistemato `pdbbind_dataset.py`. |

### Dettaglio

**Pacchetto e import**
- Tutto il codice importabile sta in `src/gmnbind/`. Dentro il pacchetto **solo import assoluti** (`from gmnbind.data.ligand import Ligand`).
- **Regola:** il pacchetto non importa mai da `scripts/`. Sono gli script a usare il pacchetto.
- Gli script definiscono `REPO_ROOT = Path(__file__).resolve().parents[1]` e risolvono tutti i percorsi della config da lì → funzionano da qualsiasi cartella.

**`scripts/build_index.py`**
- Fa tutta la pipeline che prima era nel notebook `PDB_index_processing.ipynb`: parse del file ufficiale `INDEX_general_PL.2020R1.lst` → filtri → pAffinity (−log10 in molare) → SMILES canonici dagli SDF.
- Filtri nella config (`configs/data/index_v2020*.yaml`): `max_resolution`, `exclude_nmr`, `allowed_compare` (solo `=`), `binding_types`, `exclude_comments`.
- **Verificato**: con `index_v2020.yaml` l'output coincide con il vecchio `PL_cleaned_index.csv` (4852 complessi, stessi codici, stessi pAffinity e risoluzioni). 6 SMILES scritti diversamente (3u3u, 5m3b, 5m34, 5m2q, 5m29, 6fjm): differiscono solo nella notazione stereo `/` `\`, quasi certamente stessa molecola (versione di RDKit diversa). Da confermare con InChIKey se serve.
- La nuova struttura dei dati grezzi è piatta (`generalset/{code}/`), quindi la colonna `release_year` non serve più per i percorsi ed è stata tolta dall'output.

```bash
python scripts/build_index.py --config configs/data/index_v2020.yaml
```

**`scripts/preprocess.py` + `pdbbind_dataset.py`**
- Legge `configs/data/complex_features.yaml`, genera `protein_{code}.pt` e `ligand_{code}.pt` in `data/processed/{version}/`.
- **Versionamento manuale**: il campo `version` della config decide la cartella. Alla prima esecuzione scrive `meta.json` con la fotografia della config, il commit git (con suffisso `-dirty` se c'erano modifiche non committate), le date.
  - stessa config → **riprende** (salta i `.pt` già esistenti);
  - config diversa con stessa `version` → **si blocca** e chiede di incrementare la versione.
- Alla fine aggiorna `meta.json` con `n_complexes`, `n_excluded`, `n_smiles_fallback`, `last_run_limit` e la lista `excluded` (status `excluded` = complesso senza `.pt`, con motivo; status `fallback` = ligando costruito da SMILES perché l'SDF non si leggeva).
- `meta.json` è in git (eccezione nel `.gitignore`), i `.pt` no.
- **Regola importante (anche nel README):** il controllo confronta solo la config. **Se si modifica il codice di featurizzazione (`protein.py`, `ligand.py`, `pdbbind_dataset.py`), va incrementata `version` a mano.**
- In `pdbbind_dataset.py`: `processed_dir` sovrascritto (cartella di output dalla config), `structures_dir` e `fallback_to_smiles` come parametri, registrazione di errori e fallback in `self.failures`, rimosso il vecchio `__main__`. **Logica di generazione dei grafi invariata.**

```bash
python scripts/preprocess.py                       # usa configs/data/complex_features.yaml
python scripts/preprocess.py --limit 20            # prova veloce
```

**`scripts/train.py`**
```bash
python scripts/train.py --config configs/gmn.yaml
python scripts/train.py --config configs/baseline.yaml
```
Comportamento invariato rispetto a prima (split casuale 80/20 con seed, early stopping su `val_loss`, `summary.csv` nella cartella dei log).

---

## 4. Cosa resta da fare per chiudere il refactor

In ordine. Ogni punto = un commit (o pochi). Le caselle vanno spuntate man mano.

### R1 — Pulizie veloci
- [ ] Rimuovere la cartella di prova: `git rm -r data/processed/test` (+ cancellarla dal disco).
- [ ] Rilanciare `python scripts/preprocess.py` **senza `--limit`**, con tutto committato. Ora `v1/meta.json` riporta un giro con `last_run_limit: 20` e commit `46bc823-dirty`. Rilanciato su `v1` salta tutti i `.pt` esistenti e scrive i conteggi completi. Poi committare `data/processed/v1/meta.json`.
- [ ] Le due config dell'indice scrivono **entrambe** su `data/index/index.csv` → la seconda sovrascrive la prima. Dare output diversi, es. `index_v2020_kd.csv` e `index_v2020_kd_ki.csv`, e aggiornare `complex_features.yaml` e le config degli esperimenti di conseguenza.
- [ ] Dopo il punto precedente, rimuovere `data/index/PL_cleaned_index.csv` e `data/index/parsed_index.csv`.
- [ ] Il commento in testa a `index_v2020.yaml` cita ancora `v2020_kd_cleaned_index.csv`: aggiornarlo.
- [ ] `build_index.py`: togliere il doppio `REPO_ROOT`, il `DATA_DIR` inutilizzato, il blocco di test tra virgolette dopo `main()`.
- [ ] `complex_dataset.py`: lo smoke test nel `__main__` usa ancora `data/processed` e `data/raw/PL_cleaned_index.csv` → aggiornare i percorsi o toglierlo.
- [ ] `tokenizer.py`: rimuovere `GraphPropLayer` (codice legacy non usato) e l'import di `MessagePassing`.
- [ ] Rimuovere `todo.md` (contenuto superato da questo file e dalle issue).

### R2 — nbstripout (output dei notebook fuori da git)
Non è ancora configurato (manca `.gitattributes`). Una volta per la repo:
```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
git add .gitattributes
git add --renormalize notebooks/
git commit -m "Strip notebook outputs with nbstripout"
```
**Ognuno dei due** deve poi lanciare `nbstripout --install` sul proprio clone (la configurazione del filtro sta in `.git/config`, che non è versionato). Scriverlo nel README.

### R3 — `models/base.py`: LightningModule comune
Oggi `baseline.py` e `gmn.py` duplicano identici: `_step`, `on_validation_epoch_end`, `training_step`, `validation_step`, `configure_optimizers` (incluso lo scheduler cosine con warmup) e la testa MLP di predizione.
- [ ] Creare in `base.py` una classe `BaseAffinityModel(pl.LightningModule)` con questi metodi, **copiati così come sono** (bug inclusi).
- [ ] `BaselineGNN` e `GraphMatchingNetwork` ereditano da `BaseAffinityModel` e tengono solo `__init__` (propagazione) e `forward`.
- [ ] Opzionale ma consigliato: anche la testa MLP (`Linear → [LayerNorm] → ReLU → Dropout → Linear`) costruita in `base.py`, così è garantito che sia identica nei due modelli.
- Perché conta: il paper si regge sul confronto controllato "unica differenza = propagazione". Con `base.py` è garantito dal codice, non solo dichiarato.
- Verifica: stesso numero di parametri dei modelli prima e dopo (`sum(p.numel() for p in model.parameters())`), training corto identico (vedi R8).

### R4 — Separare featurizzazione e visualizzazione
`protein.py` e `ligand.py` importano plotly, py3Dmol, matplotlib, IPython solo per le funzioni di visualizzazione. Il preprocessing non deve dipendere da librerie grafiche.
- [ ] Creare `src/gmnbind/viz.py` e spostarci: `Protein.visualize_protein_graph`, `Protein.visualize_protein_3D`, `Ligand.mol_to_nx`, `Ligand.visualize_nx3D`, `Ligand.visualize_py3Dmol`, `Ligand.plot_2d` (come funzioni che ricevono l'oggetto, es. `plot_protein_graph(protein)`), più le costanti `ATOM_PROPERTIES`/`DEFAULT_ATOM`.
- [ ] Togliere da `protein.py`/`ligand.py` gli import grafici.
- [ ] Aggiornare i notebook che usano queste funzioni (`protein_overview`, `ligand_overview`, `attention_graph`).
- ⚠️ Questo tocca `protein.py`/`ligand.py`: **non cambia i grafi**, quindi non serve una nuova versione dei dati, ma verificare con un `--limit 20` su una cartella di prova che i `.pt` siano identici (`torch.equal` su `x`, `edge_index`, `edge_attr`).

### R5 — `metrics.py`
- [ ] Creare `src/gmnbind/metrics.py` con le metriche standard del campo (stesse di PLAGCA e degli altri lavori su CASF-2016): **RMSE, MAE, Pearson R, Spearman ρ, SD, CI**.
  - SD = deviazione standard dei residui dopo una regressione lineare tra predetto e vero (definizione CASF), **non** la std degli errori.
  - CI = concordance index (frazione di coppie ordinate correttamente).
- [ ] Usarle in `base.py` a fine epoca di validazione/test (accumulando predizioni e target su tutto l'epoch, come già si fa per Pearson).
- Nota: aggiunge metriche loggate, non cambia la loss né il training.

### R6 — `splits.py` e split da file
- [ ] `src/gmnbind/data/splits.py` con:
  - `load_split(split_dir) -> {"train": [...], "val": [...], "test": [...]}` (legge `train.txt`, `val.txt`, `test.txt`, un PDB code per riga);
  - `make_cv_folds(codes, k=5, seed=47)` → scrive `fold_0.txt … fold_4.txt`;
  - (dopo, per la standardizzazione) `make_casf_split(index_df, casf_codes, n_val=1000, seed=47)`.
- [ ] Generare e committare `data/index/cv5/` (stesso seed 47 del notebook, per riprodurre la CV del paper).
- [ ] `datamodule.py`: aggiungere la possibilità di leggere uno split da file (`split: data/index/splits/...` nella config) e un `test_dataloader`. Tenere lo split casuale come default finché non si passa al nuovo protocollo, per non cambiare i risultati.
- [ ] Codici dello split senza `.pt` → saltati con un warning che dice quanti sono.

### R7 — `scripts/cross_validate.py`
Oggi la CV vive in `notebooks/experiments.ipynb` (celle con `KFold`, `run_experiment`, lista `EXPERIMENTS`).
- [ ] Portare in uno script: legge una config esperimento + `data/index/cv5/`, allena un modello per fold, salva un CSV/JSON con le metriche per fold e media ± std.
- [ ] Le 6 configurazioni del notebook (`baseline/gmn × no_reg/reg/large`) diventano file in `configs/` (es. `configs/cv/gmn_reg.yaml`).
- ⚠️ Due cose da verificare nel notebook prima di portarlo:
  1. `build_complex_dataset(..., limit=2000)`: la CV sembra fatta sui **primi 2000 complessi**, non su tutti i 4852. Controllare con quale valore sono stati prodotti i numeri del paper LIRE.
  2. Nella cella di training `best_r` è il **massimo** di Pearson su tutte le epoche, mentre `val_loss` è il minimo: possono venire da epoche diverse (ottimistico). La cella di riepilogo successiva sembra invece prendere Pearson all'epoca del miglior `val_loss`, che è il modo corretto. Nello script usare **solo** quest'ultimo.

### R8 — Test di riferimento (numeri identici prima/dopo)
Da fare prima del merge (e idealmente già adesso, prima di R3):
- [ ] Sul tag `LIRE-2026`: training corto (`limit: 200`, seed fisso, 3 epoche, `deterministic=True` nel Trainer) per baseline e GMN; salvare le loss per epoca.
- [ ] Stessa cosa su `refactor/cleanup`: le loss devono coincidere (a meno di differenze minime dovute a versioni diverse delle librerie: annotarle).
- [ ] Salvare lo script/notebook del test in `notebooks/` o `tests/` per riusarlo dopo ogni modifica.

### R9 — Documentazione
- [ ] `README.md`: installazione (sezione 1 di questo file), dove scaricare PDBbind v2020 e struttura attesa di `data/raw/`, i tre comandi della pipeline (`build_index` → `preprocess` → `train`), regola del versionamento dei dati, `nbstripout --install`. Togliere dalla sezione "Key Results" la frase "GMN reduces the MSE": la differenza era dentro la varianza tra i fold.
- [ ] `CLAUDE.md` (ignorato da git) è obsoleto: aggiornarlo o cancellarlo.

### R10 — Chiusura
- [ ] Aprire la PR `refactor/cleanup → main` (se non già aperta come draft), review dell'altro, merge.
- [ ] Proteggere `main` su GitHub (solo PR, niente push diretti) — serve che lo faccia Isabel (repo sul suo account) o che dia i permessi di admin.
- [ ] `requirements-lock.txt` con le versioni esatte dell'ambiente funzionante: `pip freeze --exclude-editable > requirements-lock.txt`.
- [ ] Archiviare o cancellare `origin/compa` dopo aver controllato che non contenga nulla di utile.

---

## 5. Pipeline a refactor finito

```
data/raw/ ──build_index.py──▶ data/index/index_*.csv ──splits.py──▶ data/index/splits/*, cv5/*
   │                                   │
   └──────────────preprocess.py────────┴──▶ data/processed/vN/*.pt + meta.json
                                                   │
                       train.py / cross_validate.py◀┘   (config: processed_dir + split + modello)
```

Ogni esperimento è descritto completamente da **dati (versione) + split + config del modello**: ogni numero del paper deve essere riproducibile con un comando.

---

## 6. Bug e problemi noti — NON correggere durante il refactor

Da aprire come issue su GitHub dopo il merge; ognuno diventa un branch + PR, così si misura l'effetto di ciascuno.

**Modello**
1. **`δ = h − W_V·c`**: in `CrossGraphAttention` la differenza confronta lo stato del nodo con una proiezione `W_V` del contesto, quindi due spazi diversi. Nel GMN originale (Li et al. 2019) la proiezione V non c'è. Togliere `W_V` o normalizzare.
2. **Confronto baseline/GMN non controllato del tutto**: la baseline non ha ReLU né LayerNorm tra i layer GINE, la GMN sì (update MLP con LayerNorm + ReLU); il numero di parametri è diverso. Pareggiare.
3. **Niente connessioni residuali** né normalizzazione tra i layer GMN → difficile salire con il numero di layer.
4. **Scheduler**: il cosine è calcolato su `estimated_stepping_batches` (cioè sulle `epochs` della config, es. 1000), ma l'early stopping si ferma molto prima → il learning rate quasi non decade. Inoltre c'è un `decay_power = 2.0` non documentato.
5. `use_amp` nella config non viene letto: `precision='bf16-mixed'` è fisso in `train.py`.

**Dati / feature**
6. **Feature continue non scalate**: B-factor e distanze entrano grezzi in `Linear(1, d)`. Standardizzare; per le distanze valutare un'espansione RBF.
7. **Fallback silenzioso su SMILES** (`fallback_to_smiles: true`): se l'SDF non si legge, il ligando viene ricostruito con ETKDG → conformazione non cristallografica, distanze diverse. Per la v2: provare il `.mol2`, altrimenti escludere.
8. **Fallback su download RCSB** per la proteina in `pdbbind_dataset.py`: stesso problema (struttura diversa da quella di PDBbind). Da togliere.
9. **Proteina intera invece del pocket** (`structure: protein`): decidere se passare a `_pocket.pdb` (standard per il confronto con lo stato dell'arte, molto più leggero) — vedi sezione 7.
10. `build_index.py`: `UNIT_TO_MOLAR` con `fillna(1)` → un'unità non prevista verrebbe trattata come già in molare senza avviso. Scartare la riga e stampare un warning.
11. `GRAPH_CONFIG` in `protein.py` è una variabile globale: il cutoff a 8 Å non è configurabile da `complex_features.yaml` (oggi quel campo è solo documentazione).

---

## 7. Dopo il refactor: standardizzazione per EMBC 2027

Scadenza full paper indicativa: **24 gennaio 2027** (da confermare sul sito ufficiale embc.embs.org/2027).

Obiettivo: tenere l'architettura (cross-attention durante il message passing) e rendere dataset e metriche confrontabili con lo stato dell'arte (PLAGCA, GIGN, GraphscoreDTA…).

- **Dataset principale**: PDBbind v2020, **Kd + Ki** (eventualmente IC50 come quasi tutti i lavori), test su **CASF-2016 core set (285 complessi)**, tolto dal training; validation ≈ 1000 complessi dal refined set. Il setup solo-Kd resta come esperimento secondario.
- **Metriche**: RMSE, MAE, Pearson, Spearman, SD, CI (R5).
- **Pocket** da `_pocket.pdb` → nuova versione dei dati (`v2`).
- **Robustezza / leakage**: sottoinsiemi di CASF filtrati per similarità (stile CoreP/CoreL di PLAGCA: identità di sequenza < 35%, Tanimoto < 0.35), **ricalcolati rispetto al nostro training v2020**; opzionale LP-PDBBind.
- **Esperimenti minimi**: baseline vs GMN su CASF con più seed; ablation concatenazione `[h ∥ c]` vs differenza `h − c`; controlli solo-ligando e solo-proteina; validazione quantitativa dell'attenzione (AUROC sui contatti reali su tutto il test set, non solo 1A08).

---

## 8. Convenzioni di lavoro

- **Branch**: uno per cosa, non per persona, vita breve: `feat/...`, `fix/...`, `exp/...`, `refactor/...`.
- **PR** piccole, review veloce dell'altro, squash merge. Prima di aprire la PR: `git pull --rebase origin main`.
- **Notebook**: solo esplorazione e figure; la logica va nei `.py`. Con nbstripout gli output non finiscono in git.
- **Config**: un file per esperimento in `configs/`, non modificare sempre lo stesso.
- **Dati**: se si tocca la featurizzazione (config **o** codice), si incrementa `version` in `complex_features.yaml`. La config dell'esperimento indica la versione esplicitamente (`processed_dir: data/processed/v2`), mai "l'ultima".
- **Fuori da git**: `data/raw/`, `.pt`, checkpoint, `lightning_logs/`. **In git**: `data/index/` (indici + split) e `data/processed/*/meta.json`.
