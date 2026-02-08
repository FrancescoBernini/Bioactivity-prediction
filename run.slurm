#!/bin/bash

#SBATCH --job-name=bioactivity_train
#SBATCH --output="outputs/%j_train.out"
#SBATCH --error="outputs/%j_train.err"
#SBATCH --partition=gpu
#SBATCH --qos=gpu
##SBATCH --partition=gpu_guest
##SBATCH --qos=gpu_guest
#SBATCH --gres=gpu:p100:1 
##SBATCH --gres=gpu:v100_hylab:1
##SBATCH --gres=gpu:a100_40g:1
##SBATCH --gres=gpu:a100_80g:1
##SBATCH --gres=gpu:l40s_fisstat:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-10:00:00

# Preparazione cartelle
mkdir -p outputs
mkdir -p results
echo "--- Job ID: $SLURM_JOB_ID ---"
echo "Nodo: $SLURM_NODELIST"

# 1. CARICAMENTO MODULI
module load miniconda3
module load cuda
module load gnu

echo "--- CHECK NVIDIA-SMI ---"
nvidia-smi
echo "----------------------"

ENV_NAME=bioactivity-env

# 2. GESTIONE AMBIENTE
if conda env list | grep -q "$ENV_NAME"; then
    echo "✅ Ambiente trovato. Attivazione..."
    source activate "$ENV_NAME"
else
    echo "📦 Creazione nuovo ambiente..."
    # Creiamo l'ambiente base SENZA xgboost (lo mettiamo dopo con pip)
    conda create -y -n "$ENV_NAME" -c conda-forge python=3.10 pandas numpy scikit-learn matplotlib seaborn joblib jupyter nbconvert
    source activate "$ENV_NAME"
fi

# Disabilita il buffering di Python
export PYTHONUNBUFFERED=1

# 3. VERIFICA E INSTALLAZIONE PACCHETTI
# Questo blocco ora gira SEMPRE, sia che l'ambiente sia nuovo o vecchio.
# Risolve il "ModuleNotFoundError" installando i pacchetti mancanti.
echo "🔧 Verifica installazione XGBoost e UMAP..."
pip install xgboost==2.0.3 umap-learn

# Variabili per evitare conflitti
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 4. ESECUZIONE
echo "🚀 Avvio main.py..."
python main.py

echo "🏁 Terminato."