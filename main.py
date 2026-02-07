# %% [markdown]
# # TODO
# 1) Capire se non fare la media degli embedding degli spettri raggruppando per inchikey (vedi ultime due risposte gemini)
# 2) **settare n° di componenti fisse di pca in modo che molecole e proteine abbiano stesso numero di componenti**
# 3) usare UMAP invece di LDA
# 4) sistemare sezione "struttura notebook"
# - magari provo anche Support vector machine anche se è dell'altro corso
# - Potrebbe aver senso rimuovere quelle molecole che hanno dev std troppo alta (se > 15 se ne toglierebbero 12 su 714)
# - fare una dimensionality reduction manuale non ha tanto senso perchè le var in input sono embedding senza quindi un significato chiaro

# %% [markdown]
# # Predizione della Bioattività di Molecole Prodotte dalle Piante
# 
# ## Obiettivo
# >Sviluppare modelli di machine learning per predire la **bioattività** di molecole prodotte dalle piante.
# 
# ## Descrizione dataset
# Il dataset è composto da 3 componenti:
# 
# ### 1) "bioactivities" 
# >Contiene l'elenco di bioattività tra molecole e proteine target. (source: [Papyrus](https://pmc.ncbi.nlm.nih.gov/articles/PMC9824924/))
# 
# - **Identificatori**
#   - `Activity_ID` (*"primary key"*): identifica la **bioattività** di una coppia "molecola-proteina"
#     - `InChIKey` (*"foreign key"*): identifica una specifica **molecola**
#     - `target_id` (*"foreign key"*): identifica una specifica **proteina**
# 
# - **Statistiche bioattività**
#   - `pchembl_value`: valori di bioattività derivanti da multiple misurazioni
#     - 2 interpretazioni euristiche possibili:
#       1) la più comune:
#          - **Active**$~~~$: $\text{pChembl}  > 5.5$
#          - **Inactive**: $\text{pChembl} <= 5.5$
#       2) alternativa:
#          - **Active**: $\text{pChembl} >= 6.0$
#          - **Borderline**: $5.0 <= \text{pChembl} < 6.0$
#          - **Inactive**: $\text{pChembl}  < 5.0$
#   - `pchembl_value_Mean`: **media** dei valori di bioattività misurati
#   - `pchembl_value_StdDev`: **deviazione standard** dei valori di bioattività misurati
#   - `pchembl_value_SEM`: **errore standard della media** dei valori di bioattività misurati
#   - `pchembl_value_N`: **numero** dei valori di bioattività misurati 
#   - `pchembl_value_Median`: **mediana** dei valori di bioattività misurati
#   - `pchembl_value_MAD`: **deviazione assoluta dalla mediana** dei valori di bioattività misurati
# 
# - **Metadati superflui**
# 
# 	
# ### 2) "mass_spectra_embeddings"
# >Contiene gli embedding delle molecole ottenuti con il tool python [Spec2Vec](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008724) a partire dagli spettri di massa.
# 
# - **Identificatori**
#   - `spectrum_id` (*"primary key"*): identifica uno specifico **spettro di una molecola**
#     - `inchikey`: identifica una specifica **molecola**
# 
# - **Embedding features**
#   - \[`0`,...,`499`\]: valori di embedding
# 
# - **Metadati superflui**
# 
# ### 3) "protein_embeddings"
# >Contiene 3 dimensionalità di embedding delle proteine ottenuti con UniRep. (source: [Papyrus](https://pmc.ncbi.nlm.nih.gov/articles/PMC9824924/))
# 1) File `protein_embeddings_UniRep64.csv`: embedding costituito da 3 componenti da 64 features/colonne, per un totale di 192 features
# 2) File `protein_embeddings_UniRep256.csv`: embedding costituito da 3 componenti da 256 features/colonne, per un totale di 768 features
# 3) File `protein_embeddings_UniRep1900.csv`: embedding costituito da 3 componenti da 1900 features/colonne, per un totale di 5700 features
# 
# - **Identificatori**
#   - `target_id` (*"primary key"*): identifica una specifica **proteina**
# 
# - **Embedding features**
#   - \[`UniRep64_AH_1`,...,`UniRep64_AH_64`,`UniRep64_FC_1`,...,`UniRep64_FC_64`,`UniRep64_FH_1`,...,`UniRep64_FH_64`\]: valori di embedding per ciascuna delle 3 componenti
#   - \[`UniRep256_AH_1`,...,`UniRep256_AH_256`,`UniRep256_FC_256`,...
#   - \[`UniRep1900_AH_1`,...,`UniRep1900_AH_1900`,`UniRep1900_FC_1900`,...
# 
# 
# ## Struttura del Notebook
# 1. Ipotesi e deduzioni
# 2. Configurazione ambiente ed import librerie
# 3. Caricamento dei dataset
# 4. Preprocessing
# 5. Training e validazione modelli
# 6. Salvataggio modelli
# 7. Conclusioni
# 
# ---

# %% [markdown]
# # 1) Ipotesi e deduzioni
# Dato che i dataset di molecole e proteine contengono rispettivamente
# - 5798 embedding di spettri di massa per 714 molecole distinte (circa 8 per molecola) e
# - 1142 embedding di proteine distinte
# 
# avere 10092 "bioattività" per 714 molecole e 1142 proteine significa che
# - ciascuna **proteina** è stata testata con circa 9 **molecole** ($10092/1142=8.83$) e
# - ciascuna **molecola** è stata testata con circa 14 **proteine** ($10092/714=14.13$)
# 
# Da ciò si evince che **si è cercato studiare il comportamento di "poche" molecole ma "a fondo"**.
# 
# Inoltre, la natura degli embedding di molecole e proteine è differente:
# - embedding **molecole**: derivati da **spettri di massa**, misure fisiche reali soggette a rumore ambientale, strumentale, umano
# - embedding **proteine**: derivati dalla **sequenza di amminoacidi** di ogni proteina che è quindi **fissa**
# 
# A fronte di ciò è stato scelto di **non aggregare** gli spettri di massa (mediandoli) in fase di join dei dataset in modo da permettere ai modelli di **distinguere** tra **rumore** e **informazione utile** a identificare la molecola.
# ___
# Visto che l'obiettivo è costruire un classificatore che possa aiutare a identificare le molecole che hanno un "qualsiasi tipo di effetto" (bioattività) quando combinate con una proteina e che quindi vale la pena analizzare, la priorità è **evitare falsi negativi** perché i falsi positivi causeranno solo una piccola perdita di tempo in laboratorio.
# 
# Per questo motivo si è scelto di affrontare tale problema tramite **classificazione** piuttosto che **regressione** in modo da poter massimizzare il valore di ***recall*** invece di minimizzare l'errore sul valore di bioattività.
# 
# Inoltre, sono stati indicati 2 diversi criteri per classificare le bioattività e nessuna delle due dà origine a classi bilanciate infatti
# - con soglia $5.5$:
#   - 3651 `active`
#   - 6441 `inactive`
# - con soglie $5.0$ e $6.0$:
#   - 2329 `active`
#   - 6044 `borderline`
#   - 1719 `inactive`
# 
# Per entrambi questi motivi è stato deciso di scegliere come unica soglia quella che **bilancia meglio le classi** cercando nell'intervallo compreso tra $5.0$ e $5.5$ inclusi: in questo modo c'è la possibilità di **bilanciare le classi** e al contempo di **sfavorire i falsi negativi** considerando come `active` anche alcuni valori `borderline`. 
# 
# Questo ha portato inoltre a scegliere la classificazione **binaria** rispetto a quella ternaria.
# ___
# Riguardo la presenza di **3 diverse dimensionalità** per gli embedding delle proteine (UniRep 64, 256 e 1900), è stato scelto di utilizzare la **dimensionalità maggiore** (UniRep 1900). 
# 
# Questa decisione si basa sull'idea che sia **preferibile partire da una rappresentazione ricca di dettagli e sfumature strutturali, di cui estrarne l'essenza più informativa tramite algoritmi di riduzione dimensionale**.
# 
# Al contrario, **partire da embedding a bassa dimensionalità comporterebbe lavorare su dati già compressi a monte, con il rischio di aver perso irreversibilmente informazioni sottili ma cruciali che nessuna tecnica successiva potrebbe recuperare**.
# ___
# Data la dimensionalità dei dataset, per la classificazione si è scelto di effettuare una feature reduction tramite
# - **PCA** (tecnica adoperata dal modello di riferimento anche se non particolarmente indicata in ambito classificazione) e
# - **LDA** (più adatto in ambito classificazione)
# 
# e di utilizzare i modelli
# - **XGBoost** (tecnica adoperata dal modello di riferimento) e
# - **Random forest** (perfetto per la classificazione)
# 
# In particolare verranno confrontate le combinazioni
# - **PCA + XGBoost** (modello di riferimento)
# - **LDA + XGBoost**
# - **PCA + Random forest**
# - **LDA + Random forest**
# ___

# %% [markdown]
# # 2) Configurazione ambiente ed import librerie

# %%
# Data manipulation
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.base import clone

# Models
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
)

# Other
import os

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# %% [markdown]
# # 3) Caricamento ed esplorazione del dataset

# %%
# Percorso base del dataset (modificare se necessario per Colab)
# Per Google Colab: montare prima il drive e modificare il path
#from google.colab import drive
#drive.mount('/content/drive')
#BASE_PATH = Path('/content/drive/MyDrive/Colab Notebooks/dataset_bioactivities')

os.makedirs('results', exist_ok=True) # Crea la cartella per l'output

# Per esecuzione locale
BASE_PATH = Path('dataset_bioactivities')

# Caricamento bioactivities
print("📊 Caricamento bioactivities...")
df_bioactivities = pd.read_csv(BASE_PATH / 'bioactivities' / 'dataset_bioactivities_negative.csv')

# Caricamento molecule embeddings
print("📊 Caricamento mass_spectra_embeddings...")
df_molecules = pd.read_csv(BASE_PATH / 'mass_spectra_embeddings' / 'ALL_GNPS_negative_ionization_DATASET.csv')

# Caricamento protein embeddings
print("📊 Caricamento protein_embeddings...")
df_proteins_1900 = pd.read_csv(BASE_PATH / 'protein_embeddings' / 'protein_embeddings_UniRep1900.csv')

print(f"\n✅ Dataset caricati con successo!")
print(f"   - Bioactivities: {df_bioactivities.shape}")
print(f"   - Molecules: {df_molecules.shape}")
print(f"   - Proteins (UniRep1900): {df_proteins_1900.shape}")

# %%
df_bioactivities

# %%
df_molecules

# %%
df_proteins_1900

# %% [markdown]
# # 4) Preprocessing

# %% [markdown]
# ## Data cleaning
# La parte di Data Cleaning è già stata effettuata infatti:
# - non ci sono dati mancanti (a parte nei metadati superflui)
# - nel dataset delle bioattività 
#   - il dominio della feature `Quality` contiene solo la qualità "**High**"
#   - il dominio della feature `Status` contiene solo il valore **reviewed**
#     - ogni misurazione è stata controllata manualmente
# - il dominio della feature `ms_level` del dataset delle molecole contiene solo il valore $2$
#   - gli embedding sono ricchi di informazione perché comprendono le singole componenti della molecola invece che solo la molecola intera (`ms_level` $= 1$)

# %% [markdown]
# ## Feature selection

# %% [markdown]
# >Rimozione **metadati superflui** e **statistiche di bioattività** superflue.

# %%
# come statistica di bioattività si prende come riferimento pchembl_value_Mean
df_bioactivities.drop(columns=[
    'Quality', 'source', 'CID', 'SMILES', 'connectivity', 'InChI', 'InChI_AuxInfo', 'accession',
    'Protein_Type', 'AID', 'doc_id', 'Year', 'all_doc_ids', 'all_years', 'type_IC50', 'type_EC50',
    'type_KD', 'type_Ki', 'relation', 'HGNC_symbol', 'UniProtID', 'Status', 'Organism', 'Classification',
    'Length', 'Sequence',
    'pchembl_value', 'pchembl_value_StdDev', 'pchembl_value_SEM',
    'pchembl_value_N', 'pchembl_value_Median', 'pchembl_value_MAD'    
], inplace=True)

df_molecules.drop(columns=[
    'backbone_inchikey', 'spectrum_id', 'smiles', 'inchi', 'instrument', 'ms_level'
], inplace=True)

# %% [markdown]
# ## Data preparation

# %% [markdown]
# >Si cerca la soglia che meglio bilancia le classi.

# %%
# Si ricerca nel range "safe" delle soglie (5.00-5.50)
pchembl_values = np.array(df_bioactivities['pchembl_value_Mean'].values)
thresholds = np.arange(5.00, 5.51, 0.01)

# Si contano gli attivi per ogni soglia (e gli inattivi per differenza)
active_counts = np.array([(pchembl_values >= t).sum() for t in thresholds])
inactive_counts = len(pchembl_values) - active_counts

# Si individua la soglia che minimizza la differenza tra le classi
best_idx = np.argmin(np.abs(active_counts - inactive_counts))
best_balancing_threshold = thresholds[best_idx]

print(f"Soglia ottimale: {best_balancing_threshold:.2f}")
print(f"active: {active_counts[best_idx]} | inactive: {inactive_counts[best_idx]}")

# %% [markdown]
# >Viene aggiunta la colonna indicante la classe per ogni bioattività secondo la soglia trovata.

# %%
df_bioactivities['bioactivity_class'] = np.where(
    df_bioactivities['pchembl_value_Mean'] >= best_balancing_threshold,
    'active',
    'inactive'
)

# %% [markdown]
# >Si effettua il merge tra i data set **verificando che le classi rimangano bilanciate** anche dopo l'aumento di record dovuto alla presenza di più spettri per singola molecola.

# %%
# Rinomino la colonna InChIKey per uniformarla
df_bioactivities.rename(columns={'InChIKey': 'inchikey'}, inplace=True)

# Merge tra dataset bioattività e spettri molecole
df_merged = df_bioactivities.merge(df_molecules, on='inchikey', how='inner')

# Merge tra bioattività e spettri molecole con embedding proteine
df_merged = df_merged.merge(df_proteins_1900, on='target_id', how='inner')

print(f"\n✅ Dataset merged:")
print(f"df_merged: {df_merged.shape}\n")

# Calcola proporzione originale
prop_orig = df_bioactivities['bioactivity_class'].value_counts(normalize=True)

# Calcola nuova proporzione
prop_new = df_merged['bioactivity_class'].value_counts(normalize=True)

print("--- BILANCIAMENTO CLASSI ---")
print(f"Originale (Active): {prop_orig.get('active', 0):.2%}")
print(f"Merged    (Active): {prop_new.get('active', 0):.2%}")

# %% [markdown]
# ## Split con StratifiedGroupKFold (k=5)
# 
# Si effettua una **Cross-Validation con k=5 fold** usando `StratifiedGroupKFold` per:
# 1. **Stratificazione**: mantenere la distribuzione di `active`/`inactive` in ogni fold
# 2. **Raggruppamento per proteina**: in questo modo si evita che il modello impari la coppia proteina-bioattività visto che alcune proteine compaiono come **sempre** `active` o **sempre** `inactive`
# 
# > **Nota**: Standardizzazione e riduzione dimensionalità vengono applicate **dentro il loop di CV** (fit su train, transform su entrambi) per evitare data leakage.

# %%
# Colonne non-feature
non_feature_cols = ['Activity_ID', 'inchikey', 'target_id', 'pchembl_value_Mean', 'bioactivity_class']

# Identifico colonne feature
feature_cols = [c for c in df_merged.columns if c not in non_feature_cols]

# Identifico indici delle colonne molecola (numeriche: '0'-'499') e proteina (UniRep*)
mol_idx = [i for i, c in enumerate(feature_cols) if c.isdigit() or (c[0] == '-' and c[1:].isdigit())]
prot_idx = [i for i, c in enumerate(feature_cols) if c.startswith('UniRep')]

# Preparo i dati per la CV
X = df_merged[feature_cols].values
y = df_merged['bioactivity_class'].values
groups = df_merged['target_id'].values

print(f"📊 Dataset preparato:")
print(f"   - {len(mol_idx)} features molecola + {len(prot_idx)} features proteina = {len(feature_cols)} totali")
print(f"   - {len(np.unique(groups))} proteine distinte")
print(f"   - {len(X)} campioni totali")

# Configurazione CV
N_SPLITS = 5
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
print(f"\n✅ StratifiedGroupKFold con {N_SPLITS} fold configurato")

# %% [markdown]
# # 5) Training e Validazione con Cross-Validation
# 
# Si addestrano e validano i modelli **XGBoost** e **Random Forest** con le seguenti tecniche di riduzione dimensionalità:
# - **PCA** (a 32, 64, 128, 256, 500 componenti, applicato separatamente su molecole e proteine)
# - **LDA** (1 componente)
# 
# **Metriche calcolate**: Accuracy, Precision, Recall, F1-score, AUC, MCC

# %%
# Setup - Mappatura manuale delle classi (active=1, inactive=0)
def encode_y(y_raw):
    """Converte le label: active → 1, inactive → 0"""
    return (y_raw == 'active').astype(int)

# Encode delle label
y_enc = encode_y(y)
print(f"Mappatura classi: inactive=0, active=1")

# CPU assegnate con fallback a 8
n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))

# Definizione modelli
models = {
    'XGBoost': xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', device='cuda', n_jobs=n_jobs),
    'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, n_jobs=n_jobs)
}

# Configurazioni PCA: numero fisso di componenti per ciascun blocco (molecole e proteine)
PCA_N_COMPONENTS = [32, 64, 128, 256, 500]

all_results = defaultdict(lambda: defaultdict(list))

def evaluate_fold(X_train, X_val, y_train, y_val, model, key):
    """Addestra e valuta un modello su un fold. pos_label=1 (active) è il default."""

    # Clona il modello base (così ogni fold parte da zero)
    clf = clone(model)

    # 1. TRAIN: Addestriamo usando la configurazione globale (GPU se disponibile)
    clf.fit(X_train, y_train)

    # 2. FIX WARNING: Se il modello è XGBoost e sta usando la GPU,
    # lo spostiamo su CPU solo per la fase di predizione.
    # Questo evita il conflitto "Input su CPU vs Modello su GPU".
    if 'XGBClassifier' in str(type(clf)) and clf.device == 'cuda':
        clf.set_params(device='cpu')
    
    # 3. PREDICT: Ora avviene senza warning
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]  # Probabilità classe 1 (active)
    
    # Calcolo metriche
    all_results[key]['Accuracy'].append(accuracy_score(y_val, y_pred))
    all_results[key]['Precision'].append(precision_score(y_val, y_pred))  # pos_label=1 default
    all_results[key]['Recall'].append(recall_score(y_val, y_pred))
    all_results[key]['F1-score'].append(f1_score(y_val, y_pred))
    all_results[key]['AUC'].append(roc_auc_score(y_val, y_proba))
    all_results[key]['MCC'].append(matthews_corrcoef(y_val, y_pred))

def apply_pca_separated(X_tr, X_val, n_components):
    """Applica PCA separato su molecole e proteine con scaling."""
    scaler_mol, scaler_prot = StandardScaler(), StandardScaler()
    # n_components viene limitato al massimo numero di feature del blocco
    n_mol = min(n_components, len(mol_idx))
    n_prot = min(n_components, len(prot_idx))
    pca_mol = PCA(n_components=n_mol, random_state=RANDOM_STATE)
    pca_prot = PCA(n_components=n_prot, random_state=RANDOM_STATE)
    
    X_tr_mol = pca_mol.fit_transform(scaler_mol.fit_transform(X_tr[:, mol_idx]))
    X_tr_prot = pca_prot.fit_transform(scaler_prot.fit_transform(X_tr[:, prot_idx]))
    X_val_mol = pca_mol.transform(scaler_mol.transform(X_val[:, mol_idx]))
    X_val_prot = pca_prot.transform(scaler_prot.transform(X_val[:, prot_idx]))
    
    scaler_final = StandardScaler()
    return scaler_final.fit_transform(np.hstack([X_tr_mol, X_tr_prot])), \
           scaler_final.transform(np.hstack([X_val_mol, X_val_prot]))

def apply_lda(X_tr, X_val, y_tr):
    """Applica LDA su tutte le feature."""
    scaler = StandardScaler()
    lda = LDA(n_components=1)
    X_train_scaled = scaler.fit_transform(X_tr)
    return lda.fit_transform(X_train_scaled, y_tr), lda.transform(scaler.transform(X_val))

# Training con Cross-Validation
print("\n" + "=" * 75)
print("🚀 TRAINING E VALIDAZIONE")
print("=" * 75)

# Lista per salvare le statistiche
fold_stats = []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_enc, groups)):
    X_tr, X_val_raw = X[train_idx], X[val_idx]
    y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

    # Calcolo statistiche del fold corrente
    train_size = len(train_idx)
    val_size = len(val_idx)
    train_active_perc = (y_tr == 1).sum() / train_size * 100
    val_active_perc = (y_val == 1).sum() / val_size * 100
    fold_stats.append((fold_idx + 1, train_size, val_size, train_active_perc, val_active_perc))
    
    # PCA con diversi numeri di componenti
    for n_comp in PCA_N_COMPONENTS:
        X_train_red, X_val_red = apply_pca_separated(X_tr, X_val_raw, n_comp)
        for model_name, model in models.items():
            key = f'pca{n_comp}_{model_name}'
            evaluate_fold(X_train_red, X_val_red, y_tr, y_val, model, key)
    
    # LDA
    X_train_red, X_val_red = apply_lda(X_tr, X_val_raw, y_tr)
    for model_name, model in models.items():
        key = f'lda_{model_name}'
        evaluate_fold(X_train_red, X_val_red, y_tr, y_val, model, key)

# Stampa risultati per ogni configurazione
print("\n📈 Risultati per configurazione:")
for key in all_results:
    recall = all_results[key]['Recall']
    print(f"   {key}: Recall = {np.mean(recall):.3f} ± {np.std(recall):.3f}")

# Tabella risultati finale
results_summary = []
for key, metrics in all_results.items():
    # Parsing chiave: 'pca32_XGBoost' → ('PCA(32)', 'XGBoost'), 'lda_RandomForest' → ('LDA', 'RandomForest')
    if key.startswith('pca'):
        underscore_pos = key.index('_')
        reduction = f"PCA({key[3:underscore_pos]})"
        model_part = key[underscore_pos+1:]
    else:
        parts = key.split('_')
        reduction = parts[0].upper()
        model_part = parts[1]
    results_summary.append({
        'Reduction': reduction,
        'Model': model_part,
        'Accuracy': f"{np.mean(metrics['Accuracy']):.3f}±{np.std(metrics['Accuracy']):.3f}",
        'Precision': f"{np.mean(metrics['Precision']):.3f}±{np.std(metrics['Precision']):.3f}",
        'Recall': f"{np.mean(metrics['Recall']):.3f}±{np.std(metrics['Recall']):.3f}",
        'F1-score': f"{np.mean(metrics['F1-score']):.3f}±{np.std(metrics['F1-score']):.3f}",
        'MCC': f"{np.mean(metrics['MCC']):.3f}±{np.std(metrics['MCC']):.3f}",
        '_recall': np.mean(metrics['Recall'])
    })

df_results = pd.DataFrame(results_summary).sort_values('_recall', ascending=False).drop(columns=['_recall'])
print("\n📊 TABELLA RISULTATI FINALE:")
print(df_results)

# Stampa della tabella dei fold
print("\n" + "="*80)
print("🧐 DETTAGLIO DISTRIBUZIONE NEI FOLD (Training Recap)")
print("="*80)
print(f"{'Fold':^5} | {'Train Size':^10} | {'Val Size':^10} | {'Train Active %':^15} | {'Val Active %':^15}")
print("-" * 75)
for row in fold_stats:
    print(f"{row[0]:^5} | {row[1]:^10} | {row[2]:^10} | {row[3]:^15.2f}% | {row[4]:^15.2f}%")
print("-" * 75)

# %% [markdown]
# # 6) Salvataggio modelli

# %%
# import joblib
# os.makedirs('models', exist_ok=True)
#
# trained_models = {}
#
# # Ri-addestramento su tutto il dataset
# X_full = df_merged[feature_cols].values
# y_full = encode_y(df_merged['bioactivity_class'].values)
#
# # PCA con diversi numeri di componenti
# for n_comp in PCA_N_COMPONENTS:
#     X_red = apply_pca_separated(X_full, X_full, n_comp)[0]  # solo il primo output (fit su tutto)
#     for model_name, model in models.items():
#         clf = clone(model).fit(X_red, y_full)
#         key = f'pca{n_comp}_{model_name}'
#         trained_models[key] = clf
#         joblib.dump(clf, f'models/{key}.joblib')
#
# # LDA
# scaler = StandardScaler()
# lda = LDA(n_components=1)
# X_red = lda.fit_transform(scaler.fit_transform(X_full), y_full)
# for model_name, model in models.items():
#     clf = clone(model).fit(X_red, y_full)
#     key = f'lda_{model_name}'
#     trained_models[key] = clf
#     joblib.dump(clf, f'models/{key}.joblib')
#
# # Salvataggio risultati
#df_results.to_csv('/content/drive/MyDrive/Colab Notebooks/results/StratifiedGroup5Fold_separatedPCAfixedDim_allSpectrums1900.csv', index=False)
df_results.to_csv('results/5-StratifiedGroupByProtein5Fold_separatedPCAfixedDim_allSpectrums1900.csv', index=False)
# print(f"✅ Salvati {len(trained_models)} modelli in 'models/' e risultati in 'results/'")

# %%
# # Caricamento ed esecuzione di un modello salvato
# MODEL_PATH = 'models'
# model_name = 'pca_RandomForest'  # Modello con miglior Recall
#
# # Carico il modello
# loaded_model = joblib.load(f'{MODEL_PATH}/{model_name}.joblib')
#
# # Esempio di predizione (richiede X_val preprocessato con la stessa pipeline)
# # y_pred = loaded_model.predict(X_val)
# # y_pred_proba = loaded_model.predict_proba(X_val)[:, 1]

# %% [markdown]
# # 7) Conclusioni

# %% [markdown]
# ---
# 
# ## Appendice A: Istruzioni per l'Esecuzione
# 
# ### Esecuzione su Google Colab
# 
# ```python
# # 1. Monta Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# 
# # 2. Copia il dataset
# !cp -r "/content/drive/MyDrive/dataset_bioactivities" "/content/"
# 
# # 3. Installa dipendenze (se necessario)
# !pip install xgboost scikit-learn pandas matplotlib seaborn
# 
# # 4. Modifica BASE_PATH nel notebook
# BASE_PATH = Path('/content/dataset_bioactivities/dataset_bioactivities')
# ```
# 
# ### Esecuzione su HPC con GPU
# 
# ```bash
# # 1. Richiedi un nodo con GPU
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# 
# # 2. Carica i moduli necessari
# module load python/3.10
# module load cuda/11.8
# 
# # 3. Attiva l'ambiente virtuale
# source venv/bin/activate
# 
# # 4. Esegui il notebook
# jupyter nbconvert --execute --to notebook main.ipynb
# ```
# 
# ### Requisiti
# - Python >= 3.8
# - scikit-learn >= 1.0
# - xgboost >= 1.5
# - pandas >= 1.3
# - numpy >= 1.21
# - matplotlib >= 3.5
# - seaborn >= 0.11


