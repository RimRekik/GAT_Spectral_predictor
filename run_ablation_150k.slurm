#!/bin/bash
#SBATCH --job-name=graphspectra_ablation_150k
#SBATCH --array=0-2
#SBATCH --account=bun@v100
#SBATCH --partition=gpu_p13
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

module purge
module load pytorch-gpu/py3/2.8.0
set -x

STRUCTURES=(atom_aa aa_only atomic_only)
STRUCTURE=${STRUCTURES[$SLURM_ARRAY_TASK_ID]}

DATA_ROOT="/lustre/fsn1/projects/rech/bun/commun/gnn/hierarchical_dataset"
mkdir -p saved_model_full

python3 -u main.py \
    --graph_structure "$STRUCTURE" \
    --activation relu \
    --max_steps 150000 \
    --eval_every 10000 \
    --num_layers 7 \
    --hidden_dim 256 \
    --model_type hierarchical_GAT \
    --lr 5e-4 \
    --save_path "saved_model_full/model_${STRUCTURE}_150k.pt" \
    --root_train "${DATA_ROOT}/train_hierarchical_no_pos_opti" \
    --root_val "${DATA_ROOT}/val_hierarchical_no_pos_opti" \
    --root_test "${DATA_ROOT}/test_hierarchical_no_pos_opti"
