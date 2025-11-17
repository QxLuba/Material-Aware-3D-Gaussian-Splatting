# 材质增强版 Relightable 3D Gaussian

本项目在官方 Relightable 3D Gaussian 的基础上，引入**材质先验**、**扩展 BRDF** 和 **空间变化材质（SVBRDF）**，用于替换第二阶段的 NeILF 训练，实现更真实、更易编辑的材质表现。本文档提供中文使用说明，方便直接对外开源。

## 项目亮点

- **材质先验**：支持金属、玻璃、织物、塑料等 8 类常见材质，可根据分类结果自适应调整约束与正则策略。
- **扩展 BRDF 模型**：集成 Cook-Torrance 与 Disney BRDF，增加 metallic、anisotropy、sheen、clearcoat 等参数，覆盖更多真实材质。
- **SVBRDF 表达**：每个高斯点学习完整材质参数，并配合平滑/一致性约束，保证局部细节与整体稳定性。
- **神经材质分类器（可选）**：内置 `PointMaterialClassifier`，可在训练过程中联合优化并按间隔保存。
- **评估与可视化**：训练脚本会导出 render/gt/normal/base_color/roughness/metallic 等图像，方便人工质检。

## 目录结构

```
material_enhancement/
├── arguments/               # 参数解析
├── gaussian_renderer/       # 渲染核心
├── scene/                   # 场景与数据接口
├── utils/                   # 工具、损失与图像处理
├── lpipsPyTorch/            # LPIPS 指标
├── brdf_models.py           # BRDF 实现
├── enhanced_gaussian_model.py
├── enhanced_renderer.py
├── material_prior.py
├── train_3dgs.py            # 阶段一：3DGS 训练
├── train_enhanced.py        # 阶段二：材质增强训练
├── gui.py                   # 可视化调试
├── scripts/                 # 示例脚本
└── README.MD                # 本文档
```

> 建议将整个 `material_enhancement/` 目录独立上传到 GitHub，同时在仓库根目录保留 `environment.yml`、`submodules/` 等依赖。

## 环境配置要求

- **操作系统**：Ubuntu 20.04 / 22.04（建议启用 CUDA 11.6+ 驱动）
- **GPU**：至少 24 GB 显存（3090/4090 等）支持 CUDA 计算
- **Python**：3.8（与官方环境一致）
- **Conda 环境名**：`m3dg`（由 `environment.yml` 创建，后续命令默认在该环境中执行）
- **依赖扩展**：`simple-knn`、`bvh`、`r3dg-rasterization`、`torch_scatter==2.1.1`、`kornia==0.6.12`

## 环境准备

1. 安装/激活 Conda 环境：
   ```bash
   conda env create --file environment.yml
   conda activate r3dg
   ```
2. 安装依赖：
   ```bash
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
   pip install torch_scatter==2.1.1 kornia==0.6.12
   pip install ./submodules/simple-knn
   pip install ./bvh
   pip install ./r3dg-rasterization
   ```

## 数据准备

沿用官方目录结构，例如：

```
datasets/
├── nerf_synthetic/lego
├── Synthetic4Relight/air_baloons
└── ...
```

`-s` 参数始终指向单个场景目录（如 `datasets/nerf_synthetic/lego`），而非父目录。

## 训练流程

### 阶段一：3D Gaussian Splatting

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

输出将包含 `chkpntXXXXX.pth` 以及 `point_cloud` 目录，为第二阶段提供初始化。

### 阶段二：材质增强

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

建议在阶段二中冻结几何与 SH（示例脚本已将对应学习率设为 0），专注优化材质参数、全局光照与可见性。

## 常用脚本

- `scripts/train_example.sh`：最简两阶段流程，可传入 `DATASET_PATH / OUTPUT_ROOT / CHECKPOINT_ITER`。
- `scripts/train_8materials.sh`：针对 Synthetic4Relight 八种材质场景，自动补齐 Stage1 并启用神经分类器。

脚本内部会自动定位 `PROJECT_ROOT`，无需关心当前工作路径。

## 参数要点

- `--brdf_model {cook_torrance, disney}`：选择 BRDF 模型。
- `--use_material_prior / --no_material_prior`：启用或关闭材质先验。
- `--use_neural_classifier`：启用神经分类器，同时可用 `--classifier_path` 载入已有模型，或通过 `--save_classifier` 周期性保存。
- `--lambda_*`：对应材质/光照/正则损失权重，可按需求微调。

## 支持的材质类型

1. Metal（金属）
2. Glass（玻璃）
3. Fabric（织物）
4. Plastic（塑料）
5. Ceramic（陶瓷）
6. Wood（木材）
7. Skin（皮肤）
8. Liquid（液体）

材质先验会为不同类型提供 metallic、roughness、specular 等范围约束，提升收敛稳定性。

## 发布到 GitHub 的建议

1. 把 `material_enhancement/` 作为仓库根目录或子模块，README.MD 即中文说明。
2. 在顶层再准备一个总 README，说明如何获取额外依赖（数据、环境文件等）。
3. 可附上训练日志、示例渲染图或预训练模型，方便他人验证。

## 引用

如使用本项目，请引用原论文：

```
@article{R3DG2023,
  author  = {Gao, Jian and Gu, Chun and Lin, Youtian and Zhu, Hao and Cao, Xun and Zhang, Li and Yao, Yao},
  title   = {Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing},
  journal = {arXiv:2311.16043},
  year    = {2023},
}
```

## 常见问题

- **能否直接进入阶段二？** 可以，只需提供已有 3DGS 检查点（`-c`），但若几何/光照偏差较大，建议重新跑阶段一。
- **依赖是否与原仓库一致？** 是，仍需安装 simple-knn、bvh、r3dg-rasterization 等扩展。
- **如何评估材质质量？** `train_enhanced.py` 已内置评估与可视化界面，可在 `output/.../eval` 中查看所有导出图像。

如有疑问或改进建议，欢迎提交 Issue / PR！

