#!/bin/bash
#SBATCH -A plgzzsn2026-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=30G
#SBATCH -t 00:30:00
#SBATCH --array=0-2
#SBATCH -o /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/scripts/outputs/array_%A_%a.log
#SBATCH -e /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/scripts/outputs/array_%A_%a.err

MAX_PER_SHAPE=50 

cd /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D

export HF_HOME="/net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/.hf_cache"

eval "$(conda shell.bash hook)"
conda activate diffusion_3d_env

python src/extract_activations.py --max_per_shape $MAX_PER_SHAPE