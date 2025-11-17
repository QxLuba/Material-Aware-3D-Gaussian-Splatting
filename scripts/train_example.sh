#!/bin/bash
# Example two-stage pipeline for Material-Enhanced 3D Gaussian Splatting

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_PATH=${1:-datasets/nerf_synthetic/lego}
OUTPUT_ROOT=${2:-output/NeRF_Syn/lego}
OUTPUT_3DGS="${OUTPUT_ROOT}/3dgs"
OUTPUT_ENHANCED="${OUTPUT_ROOT}/enhanced"
THRESHOLD_ITER=${3:-30000}

echo "=========================================="
echo "Project root : ${PROJECT_ROOT}"
echo "Dataset      : ${DATASET_PATH}"
echo "3DGS output  : ${OUTPUT_3DGS}"
echo "Enhanced out : ${OUTPUT_ENHANCED}"
echo "Checkpoint   : chkpnt${THRESHOLD_ITER}.pth"
echo "=========================================="

# Stage 1: 3DGS training
python train_3dgs.py --eval \
    -s "${DATASET_PATH}" \
    -m "${OUTPUT_3DGS}" \
    --lambda_normal_render_depth 0.01 \
    --lambda_normal_smooth 0.01 \
    --lambda_mask_entropy 0.1 \
    --save_training_vis \
    --lambda_depth_var 1e-2

# Stage 2: Material-Enhanced training
python train_enhanced.py --eval \
    -s "${DATASET_PATH}" \
    -m "${OUTPUT_ENHANCED}" \
    -c "${OUTPUT_3DGS}/chkpnt${THRESHOLD_ITER}.pth" \
    --brdf_model cook_torrance \
    --use_material_prior \
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
    --light_lr 0.001 \
    --visibility_lr 0.0025 \
    --save_training_vis \
    --save_training_vis_iteration 200

echo "Training complete!"


