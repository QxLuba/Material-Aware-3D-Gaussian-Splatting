"""
Enhanced Gaussian Model with Extended Material Parameters
Supports: metallic, anisotropy, specular, sheen, clearcoat, etc.
"""

import torch
from torch import nn
import sys
import os

# Add parent directory to path to import from original project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scene.gaussian_model import GaussianModel as BaseGaussianModel
from utils.general_utils import inverse_sigmoid


class EnhancedGaussianModel(BaseGaussianModel):
    """
    Extended Gaussian Model with full SVBRDF support
    Adds: metallic, anisotropy, specular, sheen, clearcoat, normal_offset
    """
    
    def setup_functions(self):
        super().setup_functions()
        
        # Extended material activation functions
        if self.use_pbr:
            # Metallic: [0, 1]
            self.metallic_activation = lambda x: torch.sigmoid(x)
            self.inverse_metallic_activation = lambda y: inverse_sigmoid(y)
            
            # Anisotropy: [-1, 1] (negative = more vertical, positive = more horizontal)
            self.anisotropy_activation = lambda x: torch.tanh(x)
            self.inverse_anisotropy_activation = lambda y: torch.atanh(torch.clamp(y, -0.99, 0.99))
            
            # Specular: [0, 1]
            self.specular_activation = lambda x: torch.sigmoid(x) * 0.5 + 0.5  # [0.5, 1.0]
            self.inverse_specular_activation = lambda y: inverse_sigmoid((y - 0.5) / 0.5)
            
            # Sheen: [0, 1] (for fabrics)
            self.sheen_activation = lambda x: torch.sigmoid(x)
            self.inverse_sheen_activation = lambda y: inverse_sigmoid(y)
            
            # Clearcoat: [0, 1]
            self.clearcoat_activation = lambda x: torch.sigmoid(x)
            self.inverse_clearcoat_activation = lambda y: inverse_sigmoid(y)
            
            # Clearcoat roughness: [0, 1]
            self.clearcoat_roughness_activation = lambda x: torch.sigmoid(x) * 0.3 + 0.03  # [0.03, 0.33]
            self.inverse_clearcoat_roughness_activation = lambda y: inverse_sigmoid((y - 0.03) / 0.3)
            
            # Normal offset (for normal mapping): [-0.1, 0.1] per channel
            self.normal_offset_activation = lambda x: torch.tanh(x) * 0.1
            self.inverse_normal_offset_activation = lambda y: torch.atanh(torch.clamp(y / 0.1, -0.99, 0.99))
    
    def __init__(self, sh_degree: int, render_type='render', use_extended_materials=True):
        # Initialize base class
        super().__init__(sh_degree, render_type)
        
        self.use_extended_materials = use_extended_materials and self.use_pbr
        
        if self.use_extended_materials:
            # Initialize extended material parameters
            self._metallic = torch.empty(0)
            self._anisotropy = torch.empty(0)
            self._specular = torch.empty(0)
            self._sheen = torch.empty(0)
            self._clearcoat = torch.empty(0)
            self._clearcoat_roughness = torch.empty(0)
            self._normal_offset = torch.empty(0)  # [N, 3] for normal mapping
    
    # Property getters
    @property
    def get_metallic(self):
        if not self.use_extended_materials or self._metallic.numel() == 0:
            return None
        return self.metallic_activation(self._metallic)
    
    @property
    def get_anisotropy(self):
        if not self.use_extended_materials or self._anisotropy.numel() == 0:
            return None
        return self.anisotropy_activation(self._anisotropy)
    
    @property
    def get_specular(self):
        if not self.use_extended_materials or self._specular.numel() == 0:
            return None
        return self.specular_activation(self._specular)
    
    @property
    def get_sheen(self):
        if not self.use_extended_materials or self._sheen.numel() == 0:
            return None
        return self.sheen_activation(self._sheen)
    
    @property
    def get_clearcoat(self):
        if not self.use_extended_materials or self._clearcoat.numel() == 0:
            return None
        return self.clearcoat_activation(self._clearcoat)
    
    @property
    def get_clearcoat_roughness(self):
        if not self.use_extended_materials or self._clearcoat_roughness.numel() == 0:
            return None
        return self.clearcoat_roughness_activation(self._clearcoat_roughness)
    
    @property
    def get_enhanced_normal(self):
        """Get normal with offset for normal mapping"""
        if self.use_extended_materials and self._normal_offset.shape[0] > 0:
            normal_offset = self.normal_offset_activation(self._normal_offset)
            enhanced_normal = self.get_normal + normal_offset
            return torch.nn.functional.normalize(enhanced_normal, dim=-1)
        return self.get_normal
    
    def create_from_pcd(self, pcd, spatial_lr_scale: float):
        """Initialize from point cloud with extended parameters"""
        super().create_from_pcd(pcd, spatial_lr_scale)
        
        if self.use_extended_materials:
            N = self._xyz.shape[0]
            device = self._xyz.device
            
            # Initialize extended parameters
            # Initialize metallic with small random values to allow learning
            metallic_random = torch.rand((N, 1), dtype=torch.float, device=device) * 0.1 + 0.05  # [0.05, 0.15]
            metallic_init = inverse_sigmoid(metallic_random)
            anisotropy_init = torch.zeros((N, 1), dtype=torch.float, device=device)
            specular_init = self.inverse_specular_activation(torch.ones((N, 1), dtype=torch.float, device=device) * 0.5)
            sheen_init = inverse_sigmoid(torch.zeros((N, 1), dtype=torch.float, device=device))
            clearcoat_init = inverse_sigmoid(torch.zeros((N, 1), dtype=torch.float, device=device))
            clearcoat_roughness_init = self.inverse_clearcoat_roughness_activation(
                torch.ones((N, 1), dtype=torch.float, device=device) * 0.15)
            normal_offset_init = torch.zeros((N, 3), dtype=torch.float, device=device)
            
            self._metallic = nn.Parameter(metallic_init.requires_grad_(True))
            self._anisotropy = nn.Parameter(anisotropy_init.requires_grad_(True))
            self._specular = nn.Parameter(specular_init.requires_grad_(True))
            self._sheen = nn.Parameter(sheen_init.requires_grad_(True))
            self._clearcoat = nn.Parameter(clearcoat_init.requires_grad_(True))
            self._clearcoat_roughness = nn.Parameter(clearcoat_roughness_init.requires_grad_(True))
            self._normal_offset = nn.Parameter(normal_offset_init.requires_grad_(True))
    
    def training_setup(self, training_args):
        """Setup training with extended parameters"""
        super().training_setup(training_args)
        
        if self.use_extended_materials:
            # Add extended parameters to optimizer
            l = []
            for param_group in self.optimizer.param_groups:
                l.append(param_group)
            
            # Extended material parameter learning rates (typically smaller)
            if hasattr(training_args, 'metallic_lr'):
                metallic_lr = training_args.metallic_lr
            else:
                metallic_lr = training_args.roughness_lr
            
            if hasattr(training_args, 'anisotropy_lr'):
                anisotropy_lr = training_args.anisotropy_lr
            else:
                anisotropy_lr = training_args.roughness_lr * 0.5
            
            if hasattr(training_args, 'specular_lr'):
                specular_lr = training_args.specular_lr
            else:
                specular_lr = training_args.base_color_lr * 0.5
            
            # Add new parameter groups
            l.extend([
                {'params': [self._metallic], 'lr': metallic_lr, "name": "metallic"},
                {'params': [self._anisotropy], 'lr': anisotropy_lr, "name": "anisotropy"},
                {'params': [self._specular], 'lr': specular_lr, "name": "specular"},
                {'params': [self._sheen], 'lr': specular_lr * 0.5, "name": "sheen"},
                {'params': [self._clearcoat], 'lr': specular_lr * 0.5, "name": "clearcoat"},
                {'params': [self._clearcoat_roughness], 'lr': training_args.roughness_lr, "name": "clearcoat_roughness"},
                {'params': [self._normal_offset], 'lr': training_args.normal_lr * 0.1, "name": "normal_offset"},
            ])
            
            # Recreate optimizer with extended parameters
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    def capture(self):
        """Capture model state including extended parameters"""
        captured_list = super().capture()
        
        if self.use_extended_materials:
            captured_list.extend([
                self._metallic,
                self._anisotropy,
                self._specular,
                self._sheen,
                self._clearcoat,
                self._clearcoat_roughness,
                self._normal_offset,
            ])
        
        return captured_list
    
    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        """Create model from checkpoint, handling extended parameters"""
        (model_args, first_iter) = torch.load(checkpoint_path)
        
        # Unpack base parameters (first 15)
        (self.active_sh_degree,
         self._xyz,
         self._normal,
         self._shs_dc,
         self._shs_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         weights_accum,
         xyz_gradient_accum,
         normal_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args[:15]

        self.weights_accum = weights_accum
        self.normal_gradient_accum = normal_gradient_accum
        self.denom = denom

        # Handle PBR parameters (6 parameters)
        if self.use_pbr:
            if len(model_args) > 15:
                # Check if we have extended parameters
                if len(model_args) >= 15 + 6 + 7 and self.use_extended_materials:
                    # Has extended parameters
                    (self._base_color,
                     self._roughness,
                     self._incidents_dc,
                     self._incidents_rest,
                     self._visibility_dc,
                     self._visibility_rest) = model_args[15:15+6]
                    
                    # Extended parameters
                    (self._metallic,
                     self._anisotropy,
                     self._specular,
                     self._sheen,
                     self._clearcoat,
                     self._clearcoat_roughness,
                     self._normal_offset) = model_args[15+6:15+6+7]
                elif len(model_args) >= 15 + 6:
                    # Standard PBR parameters only
                    (self._base_color,
                     self._roughness,
                     self._incidents_dc,
                     self._incidents_rest,
                     self._visibility_dc,
                     self._visibility_rest) = model_args[15:15+6]
                    # Initialize extended parameters if needed
                    if self.use_extended_materials:
                        self._initialize_extended_materials()
                else:
                    # No PBR parameters, initialize them
                    self._initialize_pbr_materials()
            else:
                # No PBR parameters at all
                self._initialize_pbr_materials()

        return first_iter
    
    def _initialize_pbr_materials(self):
        """Initialize standard PBR materials from SH colors"""
        device = self._xyz.device
        
        # Initialize base_color from SH DC component (average color)
        # SH DC shape: [N, 1, 3] (from checkpoint), we want [N, 3] for base_color
        if hasattr(self, '_shs_dc') and self._shs_dc is not None:
            # SH DC to RGB: sh_dc + 0.5 gives approximate RGB
            # Squeeze the middle dimension [N, 1, 3] -> [N, 3]
            sh_dc_rgb = (self._shs_dc.squeeze(1) + 0.5).clamp(0.0, 1.0)  # [N, 3]
            # base_color activation: sigmoid(x) * 0.77 + 0.03, range [0.03, 0.8]
            # Convert RGB to parameter space: inverse_sigmoid((rgb - 0.03) / 0.77)
            base_color_albedo = (sh_dc_rgb - 0.03) / 0.77
            base_color_albedo = base_color_albedo.clamp(0.01, 0.99)  # Clamp for numerical stability
            base_color_param = inverse_sigmoid(base_color_albedo)
            self._base_color = nn.Parameter(base_color_param.requires_grad_(True))
        else:
            # Fallback: initialize to gray (0.5 in RGB -> ~0.4 in albedo)
            gray_albedo = (torch.tensor([0.5, 0.5, 0.5], device=device) - 0.03) / 0.77
            gray_param = inverse_sigmoid(gray_albedo)
            self._base_color = nn.Parameter(gray_param.repeat(self._xyz.shape[0], 1).requires_grad_(True))
        
        roughness = torch.zeros_like(self._xyz[..., :1])
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        incidents = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2), device=device, dtype=torch.float)
        self._incidents_dc = nn.Parameter(
            incidents[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._incidents_rest = nn.Parameter(
            incidents[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        visibility = torch.zeros((self._xyz.shape[0], 1, 4 ** 2), device=device, dtype=torch.float)
        self._visibility_dc = nn.Parameter(
            visibility[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._visibility_rest = nn.Parameter(
            visibility[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
    
    def _initialize_extended_materials(self):
        """Initialize extended material parameters"""
        # Initialize metallic with small random values instead of zeros to allow learning
        # Use inverse_sigmoid to map from [0.05, 0.15] range (slightly above zero)
        metallic_init = self.inverse_metallic_activation(
            torch.rand((self._xyz.shape[0], 1), dtype=torch.float, device=self._xyz.device) * 0.1 + 0.05
        )
        self._metallic = nn.Parameter(metallic_init.requires_grad_(True))
        
        anisotropy = torch.zeros_like(self._xyz[..., :1])
        self._anisotropy = nn.Parameter(anisotropy.requires_grad_(True))
        
        specular = torch.zeros_like(self._xyz)
        self._specular = nn.Parameter(specular.requires_grad_(True))
        
        sheen = torch.zeros_like(self._xyz[..., :1])
        self._sheen = nn.Parameter(sheen.requires_grad_(True))
        
        clearcoat = torch.zeros_like(self._xyz[..., :1])
        self._clearcoat = nn.Parameter(clearcoat.requires_grad_(True))
        
        clearcoat_roughness = torch.zeros_like(self._xyz[..., :1])
        self._clearcoat_roughness = nn.Parameter(clearcoat_roughness.requires_grad_(True))
        
        normal_offset = torch.zeros_like(self._normal)
        self._normal_offset = nn.Parameter(normal_offset.requires_grad_(True))

    def restore(self, model_args, training_args, is_training=False, restore_optimizer=True):
        """Restore model state including extended parameters"""
        if self.use_extended_materials:
            # Check if extended parameters exist in checkpoint
            if len(model_args) > 22:  # Base has 15, PBR adds 6, extended adds 7 more
                extended_params = model_args[15+6:15+6+7]
                (
                    self._metallic,
                    self._anisotropy,
                    self._specular,
                    self._sheen,
                    self._clearcoat,
                    self._clearcoat_roughness,
                    self._normal_offset,
                ) = extended_params
        
        # Call parent restore
        super().restore(model_args, training_args, is_training, restore_optimizer)

