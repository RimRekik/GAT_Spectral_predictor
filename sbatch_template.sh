#!/bin/bash
#SBATCH --job-name=GraphSpectra_Run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00            # Réduit à 1 jour par run (largement suffisant pour 5000 steps)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/train_run_%j.log

cd /store/pmcs2i/AIDIBOP/GAT_Spectral_predictor

module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
source ~/venvs/graphspectra/bin/activate

echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Nœud       : $SLURM_NODELIST"
echo "Python     : $(which python)"
echo "Paramètres : LR=__LR__, DIM=__DIM__, LAYERS=__LAYERS__"
echo "========================================"

# Launch training pour CETTE combinaison précise
python main.py \
    --max_steps 10000 \
    --batch_size __BATCH_SIZE__ \
    --lr __LR__ \
    --hidden_dim __DIM__ \
    --num_layers __LAYERS__ \
    --save_path __SAVE_PATH__

echo "========================================"
echo "Terminé le : $(date)"
echo "========================================"

# Synchronisation WandB offline spécifique si nécessaire
WANDB_DIR="wandb_run/wandb"
if [ -d "$WANDB_DIR" ]; then
    wandb sync "$WANDB_DIR"/offline-run-*/
fi