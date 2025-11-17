"""
Enhanced Renderer with Material Prior and Extended BRDF
"""

import math
import torch
import numpy as np
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arguments import OptimizationParams
from scene.cameras import Camera
from utils.sh_utils import eval_sh
from utils.loss_utils import (
    ssim, first_order_edge_aware_loss, 
    tv_loss
)
from utils.image_utils import psnr
from utils.graphics_utils import fibonacci_sphere_sampling, rgb_to_srgb
from gaussian_renderer.r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# Import our new modules
from material_enhancement.brdf_models import CookTorranceBRDF, DisneyBRDF, select_brdf_model
from material_enhancement.material_prior import (
    MaterialClassifier, MaterialPriorLoss, get_material_smooth_weight,
    PointMaterialClassifier, extract_point_features
)

# Global material classifier instance (can be set from outside)
_global_material_classifier = None
_use_neural_classifier = False


def set_material_classifier(classifier: PointMaterialClassifier, use_neural: bool = True):
    """
    Set the global material classifier
    
    Args:
        classifier: PointMaterialClassifier instance or None
        use_neural: Whether to use neural classifier (True) or fallback to rule-based (False)
    """
    global _global_material_classifier, _use_neural_classifier
    _global_material_classifier = classifier
    _use_neural_classifier = use_neural
    if classifier is not None:
        classifier.eval()  # Set to evaluation mode by default


def get_material_classifier():
    """Get the current global material classifier"""
    return _global_material_classifier


def is_using_neural_classifier():
    """Check if neural classifier is enabled"""
    return _use_neural_classifier and _global_material_classifier is not None


def render_view(
    viewpoint_camera: Camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    is_training=False,
    dict_params=None,
    brdf_model_type="cook_torrance",
    use_material_prior=True
):
    """
    Enhanced render view with extended BRDF and material prior
    
    Args:
        brdf_model_type: "cook_torrance" or "disney"
        use_material_prior: Whether to use material prior classification
    """
    if dict_params is None:
        dict_params = {}
    direct_light_env_light = dict_params.get("env_light")
    
    if direct_light_env_light is None:
        raise ValueError("env_light is required in dict_params for enhanced rendering. Please ensure pbr_kwargs contains 'env_light'.")
    
    # Create zero tensor for gradients
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    intrinsic = viewpoint_camera.intrinsics
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    
    # Covariance
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # Colors/SHs
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            dir_pp_normalized = F.normalize(
                viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D, dim=-1)
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color
    
    # Get extended material parameters
    base_color = pc.get_base_color
    roughness = pc.get_roughness
    normal = pc.get_enhanced_normal if hasattr(pc, 'get_enhanced_normal') else pc.get_normal
    
    # Extended parameters (if available)
    metallic = pc.get_metallic if hasattr(pc, 'get_metallic') and pc.get_metallic is not None else None
    anisotropy = pc.get_anisotropy if hasattr(pc, 'get_anisotropy') and pc.get_anisotropy is not None else None
    specular = pc.get_specular if hasattr(pc, 'get_specular') and pc.get_specular is not None else None
    sheen = pc.get_sheen if hasattr(pc, 'get_sheen') and pc.get_sheen is not None else None
    clearcoat = pc.get_clearcoat if hasattr(pc, 'get_clearcoat') and pc.get_clearcoat is not None else None
    clearcoat_roughness = pc.get_clearcoat_roughness if hasattr(pc, 'get_clearcoat_roughness') and pc.get_clearcoat_roughness is not None else None
    
    # Default metallic if not available or ensure correct shape
    if metallic is None or metallic.numel() == 0:
        metallic = torch.zeros((base_color.shape[0], 1), device=base_color.device, dtype=base_color.dtype)
    else:
        # Ensure metallic has shape [N, 1]
        if metallic.dim() == 1:
            metallic = metallic.unsqueeze(-1)  # [N] -> [N, 1]
        elif metallic.dim() == 0:
            metallic = metallic.unsqueeze(0).unsqueeze(-1).expand(base_color.shape[0], -1)
        elif metallic.shape[0] != base_color.shape[0]:
            # If shape mismatch, create default
            metallic = torch.zeros((base_color.shape[0], 1), device=base_color.device, dtype=base_color.dtype)
        # Ensure same device
        metallic = metallic.to(base_color.device)
    
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
    incidents = pc.get_incidents
    
    # Material classification (if enabled)
    material_probs = None
    if use_material_prior and is_training:
        # Use neural network classifier if available, otherwise fallback to rule-based
        if _use_neural_classifier and _global_material_classifier is not None:
            # Extract point features for neural network
            point_features = extract_point_features(pc)
            
            # Normalize features (optional, but can help training stability)
            # Note: For now, we use raw features as they are already in reasonable ranges
            
            # Forward through network
            # Always enable gradients during training so classifier can learn
            material_probs = _global_material_classifier(point_features)
        else:
            # Fallback to rule-based classifier
            material_probs = MaterialClassifier.classify_from_params(
                base_color, roughness, metallic, normal
            )
    
    # Use precomputed incident_dirs and incident_areas (from update_visibility)
    # This matches the original neilf.py implementation
    if hasattr(pc, '_incident_dirs') and pc._incident_dirs is not None:
        incident_dirs = pc._incident_dirs
        incident_areas = pc._incident_areas
    else:
        # Fallback: compute on the fly if not precomputed
        sample_num = dict_params.get('sample_num', 64) if dict_params else 64
        if is_training:
            incident_dirs, incident_areas = fibonacci_sphere_sampling(
                normal, sample_num, random_rotate=True)
        else:
            incident_dirs, incident_areas = fibonacci_sphere_sampling(
                normal, sample_num, random_rotate=False)
    
    # Compute incident lights
    deg = int(np.sqrt(incidents.shape[1]) - 1)
    global_incident_lights = direct_light_env_light.direct_light(incident_dirs)
    local_incident_lights = eval_sh(
        deg, incidents.transpose(1, 2).view(-1, 1, 3, (deg + 1) ** 2), incident_dirs).clamp_min(0)
    
    incident_visibility = pc._visibility_tracing if hasattr(pc, '_visibility_tracing') and pc._visibility_tracing is not None else None
    if incident_visibility is not None:
        global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = local_incident_lights + global_incident_lights
    
    # Ensure incident lights are non-negative (match original neilf.py)
    # Don't add ambient light here - let the model learn proper lighting
    incident_lights = torch.clamp(incident_lights, min=0.0)
    
    # Compute BRDF
    if brdf_model_type == "disney" and specular is not None:
        # Disney BRDF
        brdf_results = DisneyBRDF.disney_brdf(
            normal, viewdirs, incident_dirs,
            base_color, roughness, metallic,
            specular=specular,
            clearcoat=clearcoat,
            clearcoat_roughness=clearcoat_roughness,
            sheen=sheen,
            anisotropy=anisotropy
        )
        f_d = brdf_results["diffuse"]
        f_s = brdf_results["specular"]
        brdf_color = brdf_results  # Store as dict for later use
    else:
        # Cook-Torrance BRDF
        # Always use CookTorrance BRDF for enhanced model (even if metallic is small)
        # This ensures consistency with training
        if metallic is not None and metallic.numel() > 0:
            # Use CookTorrance with metallic
            f_d = CookTorranceBRDF.lambert_diffuse(base_color, metallic).unsqueeze(1)  # [N, 1, 3]
            f_s = CookTorranceBRDF.cook_torrance_specular(
                normal, viewdirs, incident_dirs,
                roughness, metallic, base_color, anisotropy
            )  # [N, S, 3]
        else:
            # Fallback: simple Lambertian + GGX (for compatibility)
            f_d = base_color.unsqueeze(1) / np.pi  # [N, 1, 3]
            from gaussian_renderer.neilf import GGX_specular
            roughness_for_ggx = roughness.squeeze(-1) if roughness.dim() > 1 and roughness.shape[-1] == 1 else roughness
            f_s = GGX_specular(normal, viewdirs, incident_dirs, roughness_for_ggx, fresnel=0.04)
        
        brdf_color = f_d + f_s
        brdf_results = {"diffuse": f_d, "specular": f_s}
    
    # Compute transport
    n_d_i = (normal.unsqueeze(1) * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    transport = incident_lights * incident_areas * n_d_i  # [N, S, 3]
    
    # Render equation
    if brdf_model_type == "disney" and isinstance(brdf_results, dict):
        # Handle Disney BRDF results
        f_d = brdf_results["diffuse"]
        f_s = brdf_results["specular"]
        specular_term = (f_s * transport).mean(dim=-2)
        pbr = ((f_d + f_s) * transport).mean(dim=-2)
        if brdf_results.get("clearcoat") is not None:
            pbr = pbr + (brdf_results["clearcoat"] * transport).mean(dim=-2)
        if brdf_results.get("sheen") is not None:
            pbr = pbr + (brdf_results["sheen"] * transport).mean(dim=-2)
    else:
        # Cook-Torrance
        specular_term = (f_s * transport).mean(dim=-2) if f_s.dim() == 3 else torch.zeros_like(base_color)
        # Ensure all tensors are on the same device and have correct dimensions
        if f_d.dim() == 3 and f_s.dim() == 3:
            brdf_color_combined = f_d + f_s
            pbr = (brdf_color_combined * transport).mean(dim=-2)
        elif f_d.dim() == 3:  # [N, 1, 3]
            # Expand f_d to match f_s shape [N, S, 3]
            brdf_color_combined = f_d.expand(-1, f_s.shape[1], -1) + f_s  # [N, S, 3]
            pbr = (brdf_color_combined * transport).mean(dim=-2)
        else:
            # Both should be same shape, just add
            brdf_color_combined = f_d + f_s
            pbr = (brdf_color_combined * transport).mean(dim=-2) if brdf_color_combined.dim() == 3 else (brdf_color_combined.unsqueeze(1) * transport).mean(dim=-2)
    
    diffuse_light = transport.mean(dim=-2)
    
    # Prepare features for rasterization
    depths = (means3D - viewpoint_camera.camera_center).norm(dim=-1, keepdim=True)
    depths2 = depths.square()
    
    # Ensure pbr has correct shape [N, 3]
    if pbr.dim() == 1 and pbr.shape[0] == base_color.shape[0]:
        pbr = pbr.unsqueeze(-1).expand(-1, 3)
    elif pbr.dim() == 2 and pbr.shape[1] != 3:
        # If wrong shape, take first 3 channels
        pbr = pbr[:, :3] if pbr.shape[1] >= 3 else torch.zeros(base_color.shape[0], 3, device=base_color.device, dtype=base_color.dtype)
    
    if is_training:
        metallic_feat = metallic if metallic is not None else torch.zeros_like(roughness)
        vis_feat = incident_visibility.mean(-2) if incident_visibility is not None else torch.zeros_like(depths)
        features = torch.cat([
            depths, depths2, pbr, normal, base_color, roughness,
            metallic_feat, diffuse_light, vis_feat
        ], dim=-1)
    else:
        metallic_feat = metallic if metallic is not None else torch.zeros_like(roughness)
        specular_feat = specular_term if isinstance(specular_term, torch.Tensor) and specular_term.shape == base_color.shape else torch.zeros_like(base_color)
        vis_feat = incident_visibility.mean(-2) if incident_visibility is not None else torch.zeros_like(depths)
        features = torch.cat([
            depths, depths2, pbr, normal, base_color, roughness,
            metallic_feat, specular_feat, diffuse_light,
            incident_lights.mean(-2), vis_feat
        ], dim=-1)
    
    # Rasterize
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, weights, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )
    
    # Extract rendered features
    mask = num_contrib > 0
    # Normalize by opacity only where there are contributions
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5)
    # Apply mask to zero out areas with no contributions
    rendered_feature = rendered_feature * mask.float()
    
    # Use split like original code - split along channel dimension (dim=0 for [C, H, W])
    if is_training:
        # Training: depths(1), depths2(1), pbr(3), normal(3), base_color(3), roughness(1), metallic(1), diffuse(3), vis(1)
        # Total: 17 channels if metallic present, 16 otherwise
        if rendered_feature.shape[0] >= 17:
            split_sizes = [1, 1, 3, 3, 3, 1, 1, 3, 1]  # with metallic
            rendered_depth, rendered_depth2, rendered_pbr, rendered_normal, rendered_base_color, \
                rendered_roughness, rendered_metallic, rendered_diffuse, rendered_visibility \
                = rendered_feature.split(split_sizes, dim=0)
        else:
            split_sizes = [1, 1, 3, 3, 3, 1, 3, 1]  # without metallic
            rendered_depth, rendered_depth2, rendered_pbr, rendered_normal, rendered_base_color, \
                rendered_roughness, rendered_diffuse, rendered_visibility \
                = rendered_feature.split(split_sizes, dim=0)
            rendered_metallic = torch.zeros_like(rendered_roughness)
    else:
        # Eval: more features
        if rendered_feature.shape[0] >= 20:
            split_sizes = [1, 1, 3, 3, 3, 1, 1, 3, 3, 3, 1]  # with metallic, specular, diffuse, lights, vis
            rendered_depth, rendered_depth2, rendered_pbr, rendered_normal, rendered_base_color, \
                rendered_roughness, rendered_metallic, rendered_specular, rendered_diffuse, \
                rendered_lights, rendered_visibility \
                = rendered_feature.split(split_sizes, dim=0)
        elif rendered_feature.shape[0] >= 17:
            split_sizes = [1, 1, 3, 3, 3, 1, 3, 3, 3, 1]  # without metallic but with specular
            rendered_depth, rendered_depth2, rendered_pbr, rendered_normal, rendered_base_color, \
                rendered_roughness, rendered_specular, rendered_diffuse, rendered_lights, rendered_visibility \
                = rendered_feature.split(split_sizes, dim=0)
            rendered_metallic = torch.zeros_like(rendered_roughness)
        else:
            # Fallback: basic features only
            split_sizes = [1, 1, 3, 3, 3, 1]
            rendered_depth, rendered_depth2, rendered_pbr, rendered_normal, rendered_base_color, rendered_roughness \
                = rendered_feature.split(split_sizes, dim=0)
            rendered_metallic = torch.zeros_like(rendered_roughness)
            rendered_specular = torch.zeros_like(rendered_base_color)
    
    # Compute depth variance for regularization
    rendered_var = rendered_depth2 - rendered_depth.square()
    
    # Follow original neilf.py exactly: pbr = rendered_pbr, then composite with background
    pbr = rendered_pbr  # [C, H, W] from rasterization
    rendered_pbr_composite = pbr * rendered_opacity + (1 - rendered_opacity) * bg_color[:, None, None]
    
    # Ensure base_color is properly formatted and clamped
    # rendered_base_color is [C, H, W] from rasterization
    # Base color should already be in [0.03, 0.8] range from activation
    # First ensure it's in valid range [0, 1]
    rendered_base_color_clamped = torch.clamp(rendered_base_color, 0.0, 1.0)
    # Apply sRGB conversion for display (linear -> sRGB)
    rendered_base_color_srgb = rgb_to_srgb(rendered_base_color_clamped)
    
    # Ensure roughness and metallic are properly formatted
    # Roughness should already be in [0.09, 0.99] range from activation
    rendered_roughness_clamped = torch.clamp(rendered_roughness, 0.0, 1.0)
    rendered_metallic_clamped = torch.clamp(rendered_metallic, 0.0, 1.0) if rendered_metallic.numel() > 0 else torch.zeros_like(rendered_roughness_clamped)
    
    # Ensure normal is in [-1, 1] range
    rendered_normal_clamped = torch.clamp(rendered_normal, -1.0, 1.0)
    
    results = {
        "render": rendered_image,
        "depth": rendered_depth,
        "depth_var": rendered_var,
        "pbr": rgb_to_srgb(rendered_pbr_composite),  # Use composite version with background
        "normal": rendered_normal_clamped,
        "pseudo_normal": rendered_pseudo_normal,
        "surface_xyz": rendered_surface_xyz,
        "opacity": rendered_opacity,
        "base_color": rendered_base_color_srgb,
        "roughness": rendered_roughness_clamped,
        "metallic": rendered_metallic_clamped,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "num_rendered": num_rendered,
        "num_contrib": num_contrib,
        "diffuse_light": diffuse_light if is_training else None,
    }
    
    if not is_training:
        results["specular"] = rendered_specular if 'rendered_specular' in locals() else torch.zeros_like(rendered_base_color)
        directions = viewpoint_camera.get_world_directions()
        direct_env = direct_light_env_light.direct_light(directions.permute(1, 2, 0)).permute(2, 0, 1)
        # Follow original neilf.py exactly
        results["render_env"] = rendered_image + (1 - rendered_opacity) * rgb_to_srgb(direct_env)
        results["pbr_env"] = rgb_to_srgb(pbr * rendered_opacity + (1 - rendered_opacity) * direct_env)
    
    # Store material probabilities for loss computation
    if material_probs is not None:
        results["material_probs"] = material_probs
    
    return results


def calculate_loss(
    viewpoint_camera,
    pc,
    results,
    opt,
    direct_light_env_light,
    use_material_prior=True,
    material_prior_weight=0.1
):
    """Calculate loss with material prior constraints"""
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = results["render"]
    rendered_depth = results["depth"]
    rendered_normal = results["normal"]
    rendered_pbr = results["pbr"]
    rendered_opacity = results["opacity"]
    rendered_base_color = results["base_color"]
    rendered_roughness = results["roughness"]
    rendered_metallic = results.get("metallic", torch.zeros_like(rendered_roughness))
    
    gt_image = viewpoint_camera.original_image.cuda()
    
    # Base reconstruction loss
    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    
    # PBR reconstruction loss
    Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
    ssim_val_pbr = ssim(rendered_pbr, gt_image)
    tb_dict["l1_pbr"] = Ll1_pbr.item()
    tb_dict["ssim_pbr"] = ssim_val_pbr.item()
    tb_dict["psnr_pbr"] = psnr(rendered_pbr, gt_image).mean().item()
    loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
    loss = loss + opt.lambda_pbr * loss_pbr
    
    # Depth loss
    if opt.lambda_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        image_mask = viewpoint_camera.image_mask.cuda().bool()
        depth_mask = gt_depth > 0
        sur_mask = torch.logical_xor(image_mask, depth_mask)
        loss_depth = F.l1_loss(rendered_depth[~sur_mask], gt_depth[~sur_mask])
        tb_dict["loss_depth"] = loss_depth.item()
        loss = loss + opt.lambda_depth * loss_depth
    
    # Normal losses
    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = results['pseudo_normal']
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_normal_render_depth = F.mse_loss(
            rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    
    # Material prior loss
    if use_material_prior and "material_probs" in results:
        material_probs = results["material_probs"]
        
        # Material prior constraints - computed at Gaussian point level
        if hasattr(opt, 'lambda_material_prior') and opt.lambda_material_prior > 0:
            prior_loss_fn = MaterialPriorLoss(use_soft_constraints=True)
            
            # Get point-level material parameters
            base_color_points = pc.get_base_color  # [N, 3]
            roughness_points = pc.get_roughness  # [N, 1]
            metallic_points = pc.get_metallic if hasattr(pc, 'get_metallic') and pc.get_metallic is not None else torch.zeros_like(roughness_points)
            
            # Ensure correct shapes
            if metallic_points.dim() == 1:
                metallic_points = metallic_points.unsqueeze(-1)  # [N] -> [N, 1]
            if roughness_points.dim() == 1:
                roughness_points = roughness_points.unsqueeze(-1)  # [N] -> [N, 1]
            
            additional_params = {}
            if hasattr(pc, 'get_specular') and pc.get_specular is not None:
                specular_points = pc.get_specular
                if specular_points.dim() == 1:
                    specular_points = specular_points.unsqueeze(-1)
                additional_params["specular"] = specular_points
            
            # material_probs is already [N, num_types] at point level
            loss_material_prior = prior_loss_fn.compute_loss(
                base_color_points, roughness_points, metallic_points,
                material_probs, additional_params
            )
            tb_dict["loss_material_prior"] = loss_material_prior.item()
            loss = loss + opt.lambda_material_prior * loss_material_prior
        
        # Adaptive material-based smoothing - skip if material_probs is point-level
        # Note: For image-space smoothing, we'd need to rasterize material_probs first
        # For now, we use regular smoothing when material_probs is available
        if opt.lambda_base_color_smooth > 0:
            image_mask = viewpoint_camera.image_mask.cuda()
            # Use regular smoothing (material-based smoothing requires rasterized material_probs)
            loss_base_color_smooth = first_order_edge_aware_loss(
                rendered_base_color * image_mask, gt_image)
            tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
            loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth
    
    # Standard smoothing (if material prior not used)
    if not (use_material_prior and "material_probs" in results):
        if opt.lambda_base_color_smooth > 0:
            image_mask = viewpoint_camera.image_mask.cuda()
            loss_base_color_smooth = first_order_edge_aware_loss(
                rendered_base_color * image_mask, gt_image)
            tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
            loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth
    
    if opt.lambda_roughness_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_roughness_smooth = first_order_edge_aware_loss(
            rendered_roughness * image_mask, gt_image)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth
    
    # Normal smooth
    if opt.lambda_normal_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_normal_smooth = tv_loss(rendered_normal * image_mask)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        loss = loss + opt.lambda_normal_smooth * loss_normal_smooth
    
    # Environment smooth
    if opt.lambda_env_smooth > 0:
        env = direct_light_env_light.get_env
        loss_env_smooth = tv_loss(env[0].permute(2, 0, 1))
        tb_dict["loss_env_smooth"] = loss_env_smooth.item()
        loss = loss + opt.lambda_env_smooth * loss_env_smooth
    
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict


def render_enhanced(
    viewpoint_camera: Camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    opt: OptimizationParams = False,
    is_training=False,
    dict_params=None,
    **kwargs
):
    """
    Enhanced render function with material prior and extended BRDF
    """
    brdf_model_type = kwargs.get("brdf_model_type", "cook_torrance")
    use_material_prior = kwargs.get("use_material_prior", True)
    
    results = render_view(
        viewpoint_camera, pc, pipe, bg_color,
        scaling_modifier, override_color, is_training, dict_params,
        brdf_model_type=brdf_model_type,
        use_material_prior=use_material_prior
    )
    
    if is_training:
        loss, tb_dict = calculate_loss(
            viewpoint_camera, pc, results, opt,
            direct_light_env_light=dict_params['env_light'],
            use_material_prior=use_material_prior,
            material_prior_weight=getattr(opt, 'lambda_material_prior', 0.1)
        )
        results["tb_dict"] = tb_dict
        results["loss"] = loss
    
    return results

