# Material-Enhanced Relightable 3D Gaussian

This project extends the official Relightable 3D Gaussian with **material priors**, **extended BRDF models**, and **spatially-varying BRDF (SVBRDF)** to replace the second-stage NeILF training, achieving more realistic and editable material representations. This document provides English usage instructions for open-source release.

## Key Features

- **Material Priors**: Supports 8 common material types (metal, glass, fabric, plastic, etc.), with adaptive constraints and regularization strategies based on classification results.
- **Extended BRDF Models**: Integrates Cook-Torrance and Disney BRDF, adding parameters such as metallic, anisotropy, sheen, clearcoat, covering a wider range of realistic materials.
- **SVBRDF Representation**: Each Gaussian point learns complete material parameters, with smoothness and consistency constraints to ensure local details and global stability.
- **Neural Material Classifier (Optional)**: Built-in `PointMaterialClassifier` that can be jointly optimized during training and saved periodically.
- **Evaluation & Visualization**: Training scripts export images including render/gt/normal/base_color/roughness/metallic for manual quality inspection.

## Directory Structure

```
material_enhancement/
├── arguments/               # Parameter parsing
├── gaussian_renderer/       # Rendering core
├── scene/                   # Scene and data interfaces
├── utils/                   # Utilities, losses, and image processing
├── lpipsPyTorch/            # LPIPS metric
├── brdf_models.py           # BRDF implementations
├── enhanced_gaussian_model.py
├── enhanced_renderer.py
├── material_prior.py
├── train_3dgs.py            # Stage 1: 3DGS training
├── train_enhanced.py         # Stage 2: Material enhancement training
├── gui.py                   # Visualization and debugging
├── scripts/                 # Example scripts
├── README.md                # This document (English)
└── README.MD                # Chinese documentation
```

> It is recommended to upload the entire `material_enhancement/` directory as an independent repository to GitHub, while keeping dependencies like `environment.yml` and `submodules/` in the repository root.

## Environment Requirements

- **OS**: Ubuntu 20.04/22.04 with CUDA 11.6+ driver
- **GPU**: ≥24 GB VRAM (3090/4090 recommended) with CUDA support
- **Python**: 3.8 (matching the official repo)
- **Conda environment**: `m3dg` (created from `environment.yml`; all commands below assume it is activated)
- **Key extensions**: `simple-knn`, `bvh`, `r3dg-rasterization`, `torch_scatter==2.1.1`, `kornia==0.6.12`

## Environment Setup

1. Install/activate Conda environment:
   ```bash
   conda env create --file environment.yml
   conda activate r3dg
   ```
2. Install dependencies:
   ```bash
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
   pip install torch_scatter==2.1.1 kornia==0.6.12
   pip install ./submodules/simple-knn
   pip install ./bvh
   pip install ./r3dg-rasterization
   ```

## Data Preparation

Follow the official directory structure, for example:

```
datasets/
├── nerf_synthetic/lego
├── Synthetic4Relight/air_baloons
└── ...
```

The `-s` parameter should always point to a single scene directory (e.g., `datasets/nerf_synthetic/lego`), not the parent directory.

## Training Pipeline

### Stage 1: 3D Gaussian Splatting

```bash
python train_3dgs.py --eval \
    -s datasets/nerf_synthetic/lego \
    -m output/NeRF_Syn/lego/3dgs \
    --lambda_normal_render_depth 0.01 \
    --lambda_normal_smooth 0.01 \
    --lambda_mask_entropy 0.1 \
    --lambda_depth_var 1e-2 \
    --save_training_vis
```

The output will include `chkpntXXXXX.pth` and `point_cloud` directory, providing initialization for Stage 2.

### Stage 2: Material Enhancement

```bash
python train_enhanced.py --eval \
    -s datasets/nerf_synthetic/lego \
    -m output/NeRF_Syn/lego/enhanced \
    -c output/NeRF_Syn/lego/3dgs/chkpnt30000.pth \
    --brdf_model cook_torrance \
    --use_material_prior \
    --lambda_material_prior 0.1 \
    --lambda_base_color_smooth 0.5 \
    --lambda_roughness_smooth 0.2 \
    --iterations 40000 \
    --sample_num 64
```

It is recommended to freeze geometry and SH parameters in Stage 2 (example scripts set corresponding learning rates to 0), focusing on optimizing material parameters, global lighting, and visibility.

## Example Scripts

- `scripts/train_example.sh`: Minimal two-stage pipeline, accepts `DATASET_PATH / OUTPUT_ROOT / CHECKPOINT_ITER` as arguments.
- `scripts/train_8materials.sh`: For Synthetic4Relight scenes with 8 material types, automatically completes Stage 1 and enables neural classifier.

Scripts automatically locate `PROJECT_ROOT`, so you don't need to worry about the current working directory.

## Key Parameters

- `--brdf_model {cook_torrance, disney}`: Choose BRDF model.
- `--use_material_prior / --no_material_prior`: Enable or disable material priors.
- `--use_neural_classifier`: Enable neural classifier. You can also use `--classifier_path` to load an existing model, or `--save_classifier` to save periodically.
- `--lambda_*`: Weights for material/lighting/regularization losses, can be fine-tuned as needed.

## Supported Material Types

1. Metal
2. Glass
3. Fabric
4. Plastic
5. Ceramic
6. Wood
7. Skin
8. Liquid

Material priors provide range constraints for metallic, roughness, specular, etc., for different types, improving convergence stability.

## GitHub Release Recommendations

1. Use `material_enhancement/` as the repository root or submodule, with README.md as the English documentation.
2. Prepare a top-level README explaining how to obtain additional dependencies (data, environment files, etc.).
3. Include training logs, example renderings, or pre-trained models to facilitate verification by others.

## Citation

If you use this project, please cite the original paper:

```bibtex
@article{R3DG2023,
  author  = {Gao, Jian and Gu, Chun and Lin, Youtian and Zhu, Hao and Cao, Xun and Zhang, Li and Yao, Yao},
  title   = {Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing},
  journal = {arXiv:2311.16043},
  year    = {2023},
}
```

## FAQ

- **Can I skip Stage 1 and go directly to Stage 2?** Yes, just provide an existing 3DGS checkpoint (`-c`). However, if geometry/lighting deviations are large, it's recommended to re-run Stage 1.
- **Are dependencies the same as the original repository?** Yes, you still need to install extensions like simple-knn, bvh, r3dg-rasterization, etc.
- **How to evaluate material quality?** `train_enhanced.py` has built-in evaluation and visualization. Check all exported images in `output/.../eval`.

For questions or improvement suggestions, please submit Issues / PRs!

