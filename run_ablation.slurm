#!/bin/bash
#SBATCH --job-name=graphspectra_ablation
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=20:00:00          # TODO: ajuster selon max_steps / dataset
#SBATCH --account=bun@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10        # main.py utilise num_workers=6 pour chaque DataLoader
#SBATCH --qos=qos_gpu_a100-t3

set -euo pipefail
mkdir -p logs

module purge
module load pytorch-gpu/py3/2.8.0   # TODO: vérifier le nom exact avec `module avail pytorch-gpu` si besoin
# ou, si vous utilisez un venv/conda perso :
# source /lustre/fswork/.../venv_graphspectra/bin/activate

# -----------------------------------------------------------------
# 3 tâches du job array = 3 conditions d'ablation
# -----------------------------------------------------------------
STRUCTURES=(atom_aa aa_only atomic_only)
STRUCTURE=${STRUCTURES[$SLURM_ARRAY_TASK_ID]}

echo "=== Running ablation: $STRUCTURE (array task $SLURM_ARRAY_TASK_ID) ==="

# Racine commune vers vos datasets précalculés
DATA_ROOT="/lustre/fsn1/projects/rech/bun/commun/gnn"

# atom_aa / aa_only / atomic_only : les 3 partagent le même dataset hiérarchique,
# élagué par data/graph_structure_views.py (atomic_only garde nœuds+arêtes atome
# uniquement -- plus besoin d'un baseline_dataset séparé).
MODEL_TYPE="hierarchical_GAT"
ROOT_TRAIN="${DATA_ROOT}/hierarchical_dataset/train_hierarchical_no_pos_opti"
ROOT_VAL="${DATA_ROOT}/hierarchical_dataset/val_hierarchical_no_pos_opti"
ROOT_TEST="${DATA_ROOT}/hierarchical_dataset/test_hierarchical_no_pos_opti"

python main.py \
    --graph_structure "$STRUCTURE" \
    --model_type "$MODEL_TYPE" \
    --root_train "$ROOT_TRAIN" \
    --root_val "$ROOT_VAL" \
    --root_test "$ROOT_TEST" \
    --save_path "model/model_${STRUCTURE}.pt" \
    --max_steps 2500 \
    --eval_every 100 \
    --batch_size 512 \
    --lr 0.0001

echo "=== Done: $STRUCTURE ==="
