#!/bin/bash
#SBATCH -A plgzzsn2026-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --array=0-8
#SBATCH -o /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/scripts/outputs/probe_%A_%a.log
#SBATCH -e /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/scripts/outputs/probe_%A_%a.err

PROJECT=/net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D
EXP_DIR=$PROJECT/probe_exp/job_${SLURM_ARRAY_JOB_ID}

cd $PROJECT

eval "$(conda shell.bash hook)"
conda activate diffusion_3d_env

module load CUDA/12.4.0

python src/probing/train_probe.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --exp_dir $EXP_DIR