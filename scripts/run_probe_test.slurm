#!/bin/bash
#SBATCH -A plgzzsn2026-gpu-a100
#SBATCH -p plgrid-now
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=120G
#SBATCH -t 00:15:00
#SBATCH -o /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/scripts/outputs/probe_test_%j.log
#SBATCH -e /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/scripts/outputs/probe_test_%j.err

PROJECT=/net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D
cd $PROJECT

eval "$(conda shell.bash hook)"
conda activate /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/.conda/envs/diffusion_3d_env
module load CUDA/12.4.0

python src/probing/train_probe.py \
    --task_id 0 \
    --exp_dir $PROJECT/probe_exp/test_run