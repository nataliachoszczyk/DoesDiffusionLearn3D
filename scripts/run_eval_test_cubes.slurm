#!/bin/bash
#SBATCH -A plgzzsn2026-gpu-a100
#SBATCH -p plgrid-now
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 01:00:00
#SBATCH -o scripts/outputs/eval_test_cubes_all_%j.log
#SBATCH -e scripts/outputs/eval_test_cubes_all_%j.err

PROJECT=/net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/DoesDiffusionLearn3D
cd $PROJECT
eval "$(conda shell.bash hook)"
conda activate /net/pr2/projects/plgrid/plggzzsn2026/3d_world_in_diffusion_models/.conda/envs/diffusion_3d_env
module load CUDA/12.4.0

LAYERS=("layer_04" "layer_10" "layer_18")
TIMESTEPS=("t_200" "t_500" "t_800")

for LAYER in "${LAYERS[@]}"; do
    for TIMESTEP in "${TIMESTEPS[@]}"; do
        echo "=== $LAYER / $TIMESTEP ==="
        python src/test_cubes/eval_test_cubes.py \
            --model_dir probe_exp/pca256_job_2628839 \
            --out_dir probe_exp/test_cubes_eval_all/${LAYER}_${TIMESTEP} \
            --layer "$LAYER" \
            --timestep "$TIMESTEP"
    done
done

echo "=== All 9 combinations done ==="