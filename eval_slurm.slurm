#!/bin/bash
#SBATCH --job-name="audio_eval"
#SBATCH --partition=gpu               # Route to the GPU cluster partition
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=4             # 4 CPU cores for PyTorch DataLoader workers
#SBATCH --mem=16G                     # Request 16GB of system RAM
#SBATCH --time=01:00:00               # Walltime limit (HH:MM:SS)
#SBATCH --output=eval_out.out
#SBATCH --error=eval_err.err

module purge
module load miniforge3
conda activate dl

echo "=========================================="
echo "Job started on node: $HOSTNAME"
echo "Assigned GPU device: $CUDA_VISIBLE_DEVICES"
echo "Active Python path:  $(which python)"
echo "=========================================="

python -u eval.py