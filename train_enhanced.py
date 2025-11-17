"""
Enhanced Training Script for Material-Enhanced 3D Gaussian Splatting
Based on 3DGS training, replaces NeILF training with enhanced material model
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from collections import defaultdict
from random import randint
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.loss_utils import ssim
from scene import Scene
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from utils.system_utils import prepare_output_and_logger
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.direct_light_map import DirectLightMap
from utils.graphics_utils import rgb_to_srgb
from torchvision.utils import save_image, make_grid
from lpipsPyTorch import lpips

# Import enhanced modules
from material_enhancement.enhanced_gaussian_model import EnhancedGaussianModel
from material_enhancement.enhanced_renderer import render_enhanced, set_material_classifier
from material_enhancement.material_prior import PointMaterialClassifier


def training_enhanced(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams,
                     classifier=None, save_classifier=False, classifier_save_interval=5000, model_path=None):
    """
    Enhanced training with material prior and extended BRDF
    
    Args:
        dataset: Model parameters
        opt: Optimization parameters
        pipe: Pipeline parameters
        classifier: Optional PointMaterialClassifier instance (for neural classifier)
        save_classifier: Whether to save classifier during training
        classifier_save_interval: Interval for saving classifier
        model_path: Path to save classifier models
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # Setup Enhanced Gaussians
    gaussians = EnhancedGaussianModel(dataset.sh_degree, render_type='neilf', use_extended_materials=True)
    scene = Scene(dataset, gaussians)
    
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=False)
        # Setup training after loading checkpoint
        gaussians.training_setup(op)
    elif scene.loaded_iter:
        gaussians.load_ply(os.path.join(dataset.model_path,
                                        "point_cloud",
                                        "iteration_" + str(scene.loaded_iter),
                                        "point_cloud.ply"))
    else:
        gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)
    
    gaussians.training_setup(opt)
    
    # Setup classifier for training if provided
    if classifier is not None:
        classifier.train()  # Set to training mode for gradient flow
    
    # Setup PBR components
    pbr_kwargs = dict()
    # First update visibility
    gaussians.update_visibility(pipe.sample_num)
    
    pbr_kwargs['sample_num'] = pipe.sample_num
    print("Using global incident light for regularization.")
    direct_env_light = DirectLightMap(dataset.env_resolution, opt.light_init)
    
    if args.checkpoint:
        env_checkpoint = os.path.dirname(args.checkpoint) + "/env_light_" + os.path.basename(args.checkpoint)
        print("Trying to load global incident light from ", env_checkpoint)
        if os.path.exists(env_checkpoint):
            direct_env_light.create_from_ckpt(env_checkpoint, restore_optimizer=True)
            print("Successfully loaded!")
        else:
            print("Failed to load!")
    
    direct_env_light.training_setup(opt)
    pbr_kwargs["env_light"] = direct_env_light
    
    # Prepare render function and bg
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Training parameters
    brdf_model_type = args.brdf_model if hasattr(args, 'brdf_model') else "cook_torrance"
    use_material_prior = getattr(args, 'use_material_prior', True)
    
    # Training loop
    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress",
                        initial=first_iter, total=opt.iterations)
    
    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)
        
        # Every 1000 iterations increase SH degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        loss = 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True
        
        pbr_kwargs["iteration"] = iteration - first_iter
        render_pkg = render_enhanced(
            viewpoint_cam, gaussians, pipe, background,
            opt=opt, is_training=True, dict_params=pbr_kwargs,
            brdf_model_type=brdf_model_type,
            use_material_prior=use_material_prior
        )
        
        viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        tb_dict = render_pkg["tb_dict"]
        loss += render_pkg["loss"]
        loss.backward()
        
        # Periodic memory cleanup to avoid OOM
        if iteration % 100 == 0:
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            # Progress bar
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            pbar_dict["light_mean"] = direct_env_light.get_env.mean().item()
            pbar_dict["env"] = direct_env_light.H
            for k in tb_dict:
                if k in ["psnr", "psnr_pbr"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            progress_bar.set_postfix(pbar_dict)
            
            # Log and save
            training_report(tb_writer, iteration, tb_dict, scene, gaussians,
                          pipe, background, opt, pbr_kwargs, brdf_model_type, use_material_prior)
            
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, 
                                                    render_pkg['weights'])
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                      radii[visibility_filter])
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    densify_grad_normal_threshold = opt.densify_grad_normal_threshold if iteration > opt.normal_densify_from_iter else 99999
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,
                                              densify_grad_normal_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            # Optimizer step
            gaussians.step()
            direct_env_light.step()
            
            # Save checkpoints
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"))
                torch.save((direct_env_light.capture(), iteration),
                           os.path.join(scene.model_path, "env_light_chkpnt" + str(iteration) + ".pth"))
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
            
            # Save classifier if enabled
            if save_classifier and classifier is not None:
                if iteration % classifier_save_interval == 0 or iteration == opt.iterations:
                    classifier_path = os.path.join(scene.model_path, f"material_classifier_{iteration}.pth")
                    classifier.save(classifier_path)
                    print(f"\n[ITER {iteration}] Saved material classifier to {classifier_path}")
    
    if dataset.eval:
        # Before eval, update visibility and load latest env_light checkpoint
        print("Updating visibility for evaluation...")
        gaussians.update_visibility(pipe.sample_num)
        
        # Check for the most recent env_light checkpoint
        import glob
        env_checkpoints = glob.glob(os.path.join(scene.model_path, "env_light_chkpnt*.pth"))
        if env_checkpoints:
            # Sort by iteration number
            env_checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("env_light_chkpnt", "").replace(".pth", "")))
            latest_env_checkpoint = env_checkpoints[-1]
            print(f"Loading latest env_light checkpoint for eval: {latest_env_checkpoint}")
            direct_env_light.create_from_ckpt(latest_env_checkpoint, restore_optimizer=False)
            pbr_kwargs["env_light"] = direct_env_light
        
        eval_render(scene, gaussians, pipe, background, opt, pbr_kwargs, brdf_model_type, use_material_prior)


def training_report(tb_writer, iteration, tb_dict, scene, gaussians, pipe,
                    bg_color, opt, pbr_kwargs, brdf_model_type, use_material_prior):
    """Training report with logging"""
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f'train_loss_patches/{key}', tb_dict[key], iteration)
    
    # Report test and samples of training set
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_pbr_test = 0.0
                for idx, viewpoint in enumerate(
                        tqdm(config['cameras'], desc="Evaluating " + config['name'], leave=False)):
                    render_pkg = render_enhanced(
                        viewpoint, gaussians, pipe, bg_color,
                        opt=opt, is_training=False, dict_params=pbr_kwargs,
                        brdf_model_type=brdf_model_type,
                        use_material_prior=use_material_prior
                    )
                    
                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.cuda()
                    
                    opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
                    depth = render_pkg["depth"]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    normal = torch.clamp(
                        render_pkg.get("normal", torch.zeros_like(image)) / 2 + 0.5 * opacity, 0.0, 1.0)
                    
                    # BRDF
                    base_color = torch.clamp(render_pkg.get("base_color", torch.zeros_like(image)), 0.0, 1.0)
                    roughness = torch.clamp(render_pkg.get("roughness", torch.zeros_like(depth)), 0.0, 1.0)
                    metallic = torch.clamp(render_pkg.get("metallic", torch.zeros_like(depth)), 0.0, 1.0)
                    image_pbr = render_pkg.get("pbr", torch.zeros_like(image))
                    
                    grid = torchvision.utils.make_grid(
                        torch.stack([image, image_pbr, gt_image,
                                     opacity.repeat(3, 1, 1), depth.repeat(3, 1, 1), normal,
                                     base_color, roughness.repeat(3, 1, 1), metallic.repeat(3, 1, 1)], dim=0), nrow=3)
                    
                    if tb_writer and (idx < 2):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             grid[None], global_step=iteration)
                    
                    l1_test += F.l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()
                
                psnr_test /= len(config['cameras'])
                psnr_pbr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_PBR {}".format(iteration, config['name'], l1_test,
                                                                                    psnr_test, psnr_pbr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_pbr', psnr_pbr_test, iteration)
        
        torch.cuda.empty_cache()


def eval_render(scene, gaussians, pipe, background, opt, pbr_kwargs, brdf_model_type, use_material_prior):
    """Evaluation rendering"""
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    test_cameras = scene.getTestCameras()
    os.makedirs(os.path.join(args.model_path, 'eval', 'render'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'normal'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'base_color'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'roughness'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'metallic'), exist_ok=True)
    
    # Ensure pbr_kwargs is not None
    if pbr_kwargs is None:
        raise ValueError("pbr_kwargs cannot be None in eval_render. It must contain 'env_light'.")
    
    progress_bar = tqdm(range(0, len(test_cameras)), desc="Evaluating",
                        initial=0, total=len(test_cameras))
    
    with torch.no_grad():
        for idx in progress_bar:
            viewpoint = test_cameras[idx]
            
            # Diagnostic: check env_light
            if idx == 0:
                env_light = pbr_kwargs.get("env_light")
                if env_light is not None:
                    env_mean = env_light.get_env.mean().item()
                    env_max = env_light.get_env.max().item()
                    print(f"Env light - mean: {env_mean:.6f}, max: {env_max:.6f}")
                else:
                    print("ERROR: env_light is None in pbr_kwargs!")
            
            results = render_enhanced(
                viewpoint, gaussians, pipe, background,
                opt=opt, is_training=False, dict_params=pbr_kwargs,
                brdf_model_type=brdf_model_type,
                use_material_prior=use_material_prior
            )
            
            # Diagnostic for first image
            if idx == 0:
                pbr_raw = results.get("pbr", None)
                render_raw = results.get("render", None)
                opacity = results.get("opacity", None)
                if pbr_raw is not None:
                    print(f"PBR - min: {pbr_raw.min().item():.6f}, max: {pbr_raw.max().item():.6f}, mean: {pbr_raw.mean().item():.6f}")
                if render_raw is not None:
                    print(f"Render - min: {render_raw.min().item():.6f}, max: {render_raw.max().item():.6f}, mean: {render_raw.mean().item():.6f}")
                if opacity is not None:
                    print(f"Opacity - min: {opacity.min().item():.6f}, max: {opacity.max().item():.6f}, mean: {opacity.mean().item():.6f}")
            
            # Use PBR for evaluation (as requested)
            # Priority: pbr > pbr_env > render
            pbr_image = results.get("pbr", None)
            pbr_env_image = results.get("pbr_env", None)
            render_image = results.get("render", None)
            
            # Choose PBR if available, fallback to others
            if pbr_image is not None:
                image = pbr_image
                image_source = "pbr"
            elif pbr_env_image is not None:
                image = pbr_env_image
                image_source = "pbr_env"
            elif render_image is not None:
                image = render_image
                image_source = "render"
            else:
                print(f"ERROR: No valid image for viewpoint {idx}")
                continue
            
            # Diagnostic and validation
            image_max = image.max().item()
            image_mean = image.mean().item()
            
            if idx == 0:
                print(f"Using {image_source} - min: {image.min().item():.6f}, max: {image_max:.6f}, mean: {image_mean:.6f}")
            
            # If PBR is too dark, print diagnostic but still use it
            if image_max < 0.01 or image_mean < 0.001:
                print(f"Warning: {image_source} for viewpoint {idx} is very dark (max={image_max:.6f}, mean={image_mean:.6f})")
                if idx == 0:
                    print("Diagnostic: This may indicate incident_lights are too small or env_light needs better initialization")
            
            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
            
            # Get opacity mask for foreground/background separation
            opacity = torch.clamp(results.get("opacity", torch.ones_like(image[0:1])), 0.0, 1.0)
            if opacity.dim() == 2:
                mask = opacity.unsqueeze(0).repeat(3, 1, 1)  # [H, W] -> [3, H, W]
            elif opacity.shape[0] == 1:
                mask = opacity.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
            else:
                mask = opacity  # Already [3, H, W]
            
            bg = torch.zeros_like(image) if background is None else background.view(-1, 1, 1).expand_as(image)
            
            # Save render and GT
            save_image(image, os.path.join(args.model_path, 'eval', "render", f"{viewpoint.image_name}.png"))
            save_image(gt_image, os.path.join(args.model_path, 'eval', "gt", f"{viewpoint.image_name}.png"))
            
            # Process and save normal (convert from [-1, 1] to [0, 1])
            normal = results.get("normal", torch.zeros_like(image))
            if normal.shape[0] == 3:
                normal_save = (normal * 0.5 + 0.5) * mask + (1 - mask) * bg
            else:
                # If single channel, expand to 3 channels
                if normal.shape[0] == 1:
                    normal = normal.repeat(3, 1, 1)
                normal_save = (normal * 0.5 + 0.5) * mask + (1 - mask) * bg
            save_image(torch.clamp(normal_save, 0.0, 1.0),
                       os.path.join(args.model_path, 'eval', "normal", f"{viewpoint.image_name}.png"))
            
            # Process and save base_color
            # base_color is already sRGB converted in renderer, just clamp and mask
            base_color = results.get("base_color", torch.zeros_like(image))
            if base_color.shape[0] == 1:
                # Single channel, expand to 3 channels
                base_color = base_color.repeat(3, 1, 1)
            elif base_color.shape[0] != 3:
                # Unexpected shape, try to handle
                if base_color.dim() == 2:
                    base_color = base_color.unsqueeze(0).repeat(3, 1, 1)
                else:
                    base_color = base_color[:3]  # Take first 3 channels
            # Already sRGB converted in renderer, just clamp and apply mask
            base_color_clamped = torch.clamp(base_color, 0.0, 1.0)
            base_color_save = base_color_clamped * mask + (1 - mask) * bg
            save_image(torch.clamp(base_color_save, 0.0, 1.0),
                       os.path.join(args.model_path, 'eval', "base_color", f"{viewpoint.image_name}.png"))
            
            # Process and save roughness (single channel, expand to 3 channels)
            roughness = results.get("roughness", torch.zeros_like(opacity))
            if roughness.shape[0] == 1:
                roughness = roughness.repeat(3, 1, 1)
            elif roughness.dim() == 2:
                roughness = roughness.unsqueeze(0).repeat(3, 1, 1)
            elif roughness.shape[0] != 3:
                roughness = roughness[:1].repeat(3, 1, 1)
            roughness_save = torch.clamp(roughness, 0.0, 1.0) * mask + (1 - mask) * bg
            save_image(torch.clamp(roughness_save, 0.0, 1.0),
                       os.path.join(args.model_path, 'eval', "roughness", f"{viewpoint.image_name}.png"))
            
            # Process and save metallic (single channel, expand to 3 channels)
            metallic = results.get("metallic", torch.zeros_like(opacity))
            
            # Diagnostic: check metallic values
            if idx == 0:  # Only print for first image
                metallic_min = metallic.min().item() if metallic.numel() > 0 else 0.0
                metallic_max = metallic.max().item() if metallic.numel() > 0 else 0.0
                metallic_mean = metallic.mean().item() if metallic.numel() > 0 else 0.0
                print(f"Metallic values - min: {metallic_min:.6f}, max: {metallic_max:.6f}, mean: {metallic_mean:.6f}")
            
            if metallic.shape[0] == 1:
                metallic = metallic.repeat(3, 1, 1)
            elif metallic.dim() == 2:
                metallic = metallic.unsqueeze(0).repeat(3, 1, 1)
            elif metallic.shape[0] != 3:
                metallic = metallic[:1].repeat(3, 1, 1)
            
            # Ensure metallic is visible even if small: scale to [0, 1] range
            if metallic.max() > 0:
                metallic_normalized = metallic / metallic.max().clamp_min(1e-6)
            else:
                metallic_normalized = metallic
            
            metallic_save = torch.clamp(metallic_normalized, 0.0, 1.0) * mask + (1 - mask) * bg
            save_image(torch.clamp(metallic_save, 0.0, 1.0),
                       os.path.join(args.model_path, 'eval', "metallic", f"{viewpoint.image_name}.png"))
    
    psnr_test /= len(test_cameras)
    ssim_test /= len(test_cameras)
    lpips_test /= len(test_cameras)
    with open(os.path.join(args.model_path, 'eval', "eval.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")
    print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {}".format(args.iterations, "test", psnr_test, ssim_test,
                                                                       lpips_test))


if __name__ == "__main__":
    parser = ArgumentParser(description="Enhanced Material Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--brdf_model", type=str, default="cook_torrance", choices=["cook_torrance", "disney"])
    parser.add_argument("--use_material_prior", action='store_true', default=True)
    parser.add_argument("--no_material_prior", dest='use_material_prior', action='store_false')
    
    # Add material prior loss weight
    parser.add_argument("--lambda_material_prior", type=float, default=0.1)
    
    # Material classifier options
    parser.add_argument("--use_neural_classifier", action='store_true', default=False,
                       help="Use neural network classifier instead of rule-based classifier")
    parser.add_argument("--classifier_path", type=str, default=None,
                       help="Path to load pretrained material classifier model")
    parser.add_argument("--save_classifier", action='store_true', default=False,
                       help="Save classifier model during training")
    parser.add_argument("--classifier_save_interval", type=int, default=5000,
                       help="Interval for saving classifier model")
    
    args = parser.parse_args(sys.argv[1:])
    
    # Add material prior lambda to optimization params
    op.lambda_material_prior = args.lambda_material_prior
    
    print(f"Current model path: {args.model_path}")
    print(f"BRDF Model: {args.brdf_model}")
    print(f"Material Prior: {args.use_material_prior}")
    print(f"Neural Classifier: {args.use_neural_classifier}")
    print("Optimizing " + args.model_path)
    
    # Initialize material classifier if using neural network
    classifier = None
    if args.use_neural_classifier:
        if args.classifier_path and os.path.exists(args.classifier_path):
            print(f"Loading material classifier from {args.classifier_path}")
            classifier = PointMaterialClassifier.load(args.classifier_path, device='cuda')
        else:
            print("Initializing new material classifier")
            classifier = PointMaterialClassifier(
                feature_dim=12,  # xyz(3) + base_color(3) + roughness(1) + metallic(1) + normal(3) + opacity(1)
                num_materials=8,  # metal, glass, fabric, plastic, ceramic, wood, skin, liquid
                hidden_dim=128,
                num_layers=3
            ).cuda()
            # Start in training mode so gradients flow through it
            classifier.train()
        
        # Set as global classifier
        set_material_classifier(classifier, use_neural=True)
    else:
        # Use rule-based classifier
        set_material_classifier(None, use_neural=False)
    
    # Initialize system state
    safe_state(args.quiet)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training_enhanced(lp.extract(args), op.extract(args), pp.extract(args), 
                     classifier=classifier if args.use_neural_classifier else None,
                     save_classifier=args.save_classifier,
                     classifier_save_interval=args.classifier_save_interval,
                     model_path=args.model_path)
    
    print("\nTraining complete.")

