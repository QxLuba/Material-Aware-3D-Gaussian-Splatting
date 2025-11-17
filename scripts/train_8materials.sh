#!/bin/bash
# Training script for Material-Enhanced 3D Gaussian Splatting with 8 material types
# Materials: metal, glass, fabric, plastic, ceramic, wood, skin, liquid

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Dataset path - override with first argument if needed
DATASET_PATH=${1:-datasets/Synthetic4Relight/air_baloons}
OUTPUT_ROOT=${2:-output/Syn4Relight/air_baloons}
OUTPUT_3DGS="${OUTPUT_ROOT}/3dgs"
OUTPUT_ENHANCED="${OUTPUT_ROOT}/enhanced_8materials"
CHECKPOINT_ITER=${3:-30000}
CHECKPOINT_PATH="${OUTPUT_3DGS}/chkpnt${CHECKPOINT_ITER}.pth"

echo "=========================================="
echo "Training with 8 Material Types"
echo "Materials: metal, glass, fabric, plastic, ceramic, wood, skin, liquid"
echo "=========================================="
echo "Dataset      : ${DATASET_PATH}"
echo "3DGS output  : ${OUTPUT_3DGS}"
echo "Enhanced out : ${OUTPUT_ENHANCED}"
echo "Checkpoint   : ${CHECKPOINT_PATH}"
echo "=========================================="

# Stage 1: 3DGS training (if not already done)
# Skip this if you already have a trained 3DGS model
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Stage 1: Training 3DGS..."
    python train_3dgs.py --eval \
        -s "${DATASET_PATH}" \
        -m "${OUTPUT_3DGS}" \
        --lambda_normal_render_depth 0.01 \
        --lambda_normal_smooth 0.01 \
        --lambda_mask_entropy 0.1 \
        --save_training_vis \
        --lambda_depth_var 1e-2
else
    echo "Stage 1: 3DGS model already exists, skipping..."
fi

# Stage 2: Material-Enhanced training with 8 material types
echo "Stage 2: Training Material-Enhanced model with 8 material types..."
python train_enhanced.py --eval \
    -s "${DATASET_PATH}" \
    -m "${OUTPUT_ENHANCED}" \
    -c "${CHECKPOINT_PATH}" \
    --brdf_model cook_torrance \
    --use_material_prior \
    --use_neural_classifier \
    --lambda_material_prior 0.1 \
    --lambda_base_color_smooth 0.5 \
    --lambda_roughness_smooth 0.2 \
    --lambda_normal_smooth 0.01 \
    --lambda_pbr 1 \
    --lambda_dssim 0.2 \
    --lambda_env_smooth 0.01 \
    --iterations 40000 \
    --sample_num 64 \
    --position_lr_init 0 \
    --position_lr_final 0 \
    --normal_lr 0 \
    --sh_lr 0 \
    --opacity_lr 0 \
    --scaling_lr 0 \
    --rotation_lr 0 \
    --base_color_lr 0.01 \
    --roughness_lr 0.01 \
    --metallic_lr 0.01 \
    --light_lr 0.001 \
    --visibility_lr 0.0025 \
    --save_training_vis \
    --save_training_vis_iteration 200 \
    --save_classifier \
    --classifier_save_interval 5000

echo "=========================================="
echo "Training complete!"
echo "Model saved to: ${OUTPUT_ENHANCED}"
echo "=========================================="

