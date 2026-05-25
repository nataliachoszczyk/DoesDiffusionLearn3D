#!/bin/bash
#SBATCH -A plgzzsn2026-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=30G
#SBATCH -t 00:15:00
#SBATCH -o slurm_test_%j.log
#SBATCH -e slurm_test_%j.err

# Navigate to the project root
cd /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D

# Redirect Hugging Face cache to the project directory (large quota)
export HF_HOME="/net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/.hf_cache"

# Initialize Conda for the bash script execution
eval "$(conda shell.bash hook)"
conda activate diffusion_3d_env

# Run the test extraction script
python tests/test_extraction.py