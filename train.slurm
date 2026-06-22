#!/bin/bash
#SBATCH --job-name=graphspectra_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_6000:1      # 1 GPU (sur les 2 disponibles)
#SBATCH --cpus-per-task=8                 # 8 CPUs : 6 pour num_workers + 2 pour le process principal
#SBATCH --mem=80G                         # 80 GB RAM (largement suffisant, nœud a 190 GB)
#SBATCH --time=3-00:00:00                 # 3 jours max (limite partition gpu = 4 jours)
#SBATCH --output=logs/train_%j.log        # stdout → logs/train_<job_id>.log
#SBATCH --error=logs/train_%j.log         # stderr dans le même fichier
#SBATCH --mail-type=END,FAIL              # email quand le job finit ou crashe
#SBATCH --mail-user= rim.rekik-dit-nekhili@ec-lyon.fr  # à adapter si besoin

# -------------------------------------------------------
# Environnement
# -------------------------------------------------------
cd /store/pmcs2i/AIDIBOP/GraphSpectra

conda activate grph

# -------------------------------------------------------
# Lancement
# -------------------------------------------------------
echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Nœud       : $SLURM_NODELIST"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "Démarré le : $(date)"
echo "========================================"

python main.py \
    --max_steps 10000 \
    --eval_every 300 \
    --batch_size 512 \
    --lr 0.001 \
    --hidden_dim 128 \
    --num_layers 3 \
    --save_path model/model.pt

echo "========================================"
echo "Terminé le : $(date)"
echo "========================================"
