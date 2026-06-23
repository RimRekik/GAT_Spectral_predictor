#SBATCH --job-name=GraphSpectra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/train_%j.log

cd /store/pmcs2i/AIDIBOP/GraphSpectra
module purge
module load GCC/12.3.0
module load Python/3.10.8-GCCcore-12.2.0-bare
source ~/venvs/graphspectra/bin/activate

echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Nœud       : $SLURM_NODELIST"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "Python     : $(which python)"
echo "Démarré le : $(date)"
echo "========================================"

nvidia-smi

python -c "
import torch
print('CUDA dispo:', torch.cuda.is_available())
print('Torch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('Nb GPUs:', torch.cuda.device_count())
"
python -c "import torch_geometric; print('PyG OK')"

python main.py \
    --max_steps 10000 \
    --eval_every 300 \
    --batch_size 512 \
    --lr 0.001 \
    --hidden_dim 256 \
    --num_layers 3 \
    --save_path model/model.pt

echo "========================================"
echo "Terminé le : $(date)"
echo "========================================"

# WandB sync
WANDB_DIR="wandb_run/wandb"
if [ -d "$WANDB_DIR" ]; then
    for run_dir in "$WANDB_DIR"/offline-run-*/; do
        if [ -d "$run_dir" ]; then
            echo "Syncing $run_dir"
            wandb sync "$run_dir"
        fi
    done
else
    echo "No wandb offline runs found"
fi