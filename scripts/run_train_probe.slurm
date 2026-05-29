#!/bin/bash
#SBATCH -A plgzzsn2026-gpu-a100
#SBATCH -p plgrid-now
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 01:00:00
#SBATCH -o /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/scripts/outputs/probe_%j.log
#SBATCH -e /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D/scripts/outputs/probe_%j.err

N_PCA=1024

PROJECT=/net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D
EXP_DIR=$PROJECT/probe_exp/pca${N_PCA}_job_${SLURM_JOB_ID}

cd $PROJECT

eval "$(conda shell.bash hook)"
conda activate /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/.conda/envs/diffusion_3d_env
module load CUDA/12.4.0

for TASK_ID in $(seq 0 8); do
    python src/probing/train_probe.py \
        --task_id $TASK_ID \
        --exp_dir $EXP_DIR \
        --n_pca $N_PCA
done