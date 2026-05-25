#!/bin/bash
#SBATCH -A plgzzsn2026-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=30G
#SBATCH -t 00:30:00
#SBATCH --array=0-9
#SBATCH -o /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/scripts/outputs/fix_%A_%a.log
#SBATCH -e /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/scripts/outputs/fix_%A_%a.err

cd /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D
export HF_HOME="/net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/.hf_cache"

eval "$(conda shell.bash hook)"
conda activate diffusion_3d_env

# Uruchamiamy naprawę: tylko warstwa 18, tylko czas 200, i wymuszamy nadpisywanie plików
python src/extract_activations.py \
    --max_per_shape 1000 \
    --overwrite \
    --layers transformer_blocks.18 \
    --timesteps 200