"""
Material Prior Classification and Constraints
Provides material type classification and physics-based constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


# Material type definitions
MATERIAL_TYPES = {
    "metal": 0,
    "glass": 1,
    "fabric": 2,
    "plastic": 3,
    "ceramic": 4,
    "wood": 5,
    "skin": 6,
    "liquid": 7,
}

# Material-specific parameter ranges (normalized [0, 1])
MATERIAL_PRIORS = {
    "metal": {
        "metallic": (0.8, 1.0),
        "roughness": (0.0, 0.3),
        "specular": (0.5, 1.0),
        "base_color": "variable",
        "anisotropy": (0.0, 0.5),  # Metals can have anisotropic reflections
    },
    "glass": {
        "metallic": (0.0, 0.1),
        "roughness": (0.0, 0.2),
        "specular": (0.9, 1.0),
        "base_color": (0.9, 1.0),  # Usually close to white
        "ior": (1.4, 1.6),  # Index of refraction (if supported)
    },
    "fabric": {
        "metallic": (0.0, 0.05),
        "roughness": (0.5, 1.0),
        "specular": (0.0, 0.3),
        "base_color": "variable",
        "sheen": (0.0, 0.5),  # Fabric has sheen
    },
    "plastic": {
        "metallic": (0.0, 0.1),
        "roughness": (0.2, 0.6),
        "specular": (0.3, 0.7),
        "base_color": "variable",
        "clearcoat": (0.0, 0.3),
    },
    "ceramic": {
        "metallic": (0.0, 0.05),
        "roughness": (0.1, 0.4),
        "specular": (0.4, 0.8),
        "base_color": "variable",
    },
    "wood": {
        "metallic": (0.0, 0.05),
        "roughness": (0.4, 0.8),
        "specular": (0.1, 0.4),
        "base_color": "variable",
        "anisotropy": (0.2, 0.6),  # Wood grain has anisotropy
    },
    "skin": {
        "metallic": (0.0, 0.05),
        "roughness": (0.3, 0.6),
        "specular": (0.2, 0.5),
        "base_color": "variable",
        "subsurface": (0.3, 0.7),  # Skin has subsurface scattering
    },
    "liquid": {
        "metallic": (0.0, 0.1),
        "roughness": (0.0, 0.15),
        "specular": (0.8, 1.0),
        "base_color": "variable",
        "ior": (1.3, 1.5),  # Index of refraction for liquids
        "transmission": (0.5, 1.0),  # Liquids have transmission
    },
}


class PointMaterialClassifier(nn.Module):
    """
    Lightweight neural network for material classification based on point features.
    Works directly on Gaussian point-level features.
    """
    
    def __init__(self, feature_dim=12, num_materials=8, hidden_dim=128, num_layers=3):
        """
        Args:
            feature_dim: Input feature dimension (default: 12 for xyz(3) + base_color(3) + 
                         roughness(1) + metallic(1) + normal(3) + opacity(1))
            num_materials: Number of material types (8: metal, glass, fabric, plastic, ceramic, wood, skin, liquid)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_materials = num_materials
        self.hidden_dim = hidden_dim
        
        # Build MLP layers
        layers = []
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_dim))  # Use LayerNorm instead of BatchNorm for variable batch sizes
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))  # Use LayerNorm instead of BatchNorm
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, num_materials))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            point_features: [N, feature_dim] point features
            
        Returns:
            [N, num_materials] material type probabilities (after softmax)
        """
        logits = self.net(point_features)  # [N, num_materials]
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def save(self, path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'feature_dim': self.feature_dim,
            'num_materials': self.num_materials,
            'hidden_dim': self.hidden_dim,
        }, path)
    
    @staticmethod
    def load(path: str, device='cuda'):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=device)
        model = PointMaterialClassifier(
            feature_dim=checkpoint['feature_dim'],
            num_materials=checkpoint['num_materials'],
            hidden_dim=checkpoint['hidden_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model


def extract_point_features(pc) -> torch.Tensor:
    """
    Extract point-level features from Gaussian model for material classification
    
    Args:
        pc: GaussianModel or EnhancedGaussianModel instance
        
    Returns:
        [N, feature_dim] tensor of point features
        Feature composition: xyz(3) + base_color(3) + roughness(1) + metallic(1) + 
                             normal(3) + opacity(1) = 12 dimensions
    """
    device = pc.get_xyz.device
    
    # Extract all available features
    features_list = []
    
    # 1. 3D position [N, 3]
    xyz = pc.get_xyz
    features_list.append(xyz)
    
    # 2. Base color [N, 3]
    base_color = pc.get_base_color
    features_list.append(base_color)
    
    # 3. Roughness [N, 1]
    roughness = pc.get_roughness
    if roughness.dim() == 1:
        roughness = roughness.unsqueeze(-1)
    features_list.append(roughness)
    
    # 4. Metallic [N, 1] (if available)
    if hasattr(pc, 'get_metallic') and pc.get_metallic is not None:
        metallic = pc.get_metallic
        if metallic.dim() == 1:
            metallic = metallic.unsqueeze(-1)
    else:
        metallic = torch.zeros_like(roughness)
    features_list.append(metallic)
    
    # 5. Normal [N, 3]
    normal = pc.get_normal
    features_list.append(normal)
    
    # 6. Opacity [N, 1]
    opacity = pc.get_opacity
    if opacity.dim() == 1:
        opacity = opacity.unsqueeze(-1)
    features_list.append(opacity)
    
    # Concatenate all features
    features = torch.cat(features_list, dim=-1)  # [N, 12]
    
    return features


class MaterialClassifier:
    """Simple rule-based material classifier (legacy method)"""
    
    @staticmethod
    def classify_from_params(
        base_color: torch.Tensor,
        roughness: torch.Tensor,
        metallic: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Classify material type based on current parameters
        
        Args:
            base_color: [N, 3] base color
            roughness: [N, 1] roughness
            metallic: [N, 1] metallic (optional)
            normal: [N, 3] normal (optional)
            
        Returns:
            Material type probabilities [N, num_types]
        """
        N = base_color.shape[0]
        device = base_color.device
        
        # Ensure all inputs are on the same device
        roughness = roughness.to(device)
        if metallic is not None:
            metallic = metallic.to(device)
        if normal is not None:
            normal = normal.to(device)
        
        # Initialize probabilities
        probs = torch.zeros((N, len(MATERIAL_TYPES)), device=device, dtype=base_color.dtype)
        
        # Compute mean values for classification
        roughness_mean = roughness.mean(dim=-1, keepdim=True)  # [N, 1]
        
        if metallic is not None:
            metallic_mean = metallic.mean(dim=-1, keepdim=True)  # [N, 1]
            
            # Metal classification
            metal_mask = (metallic_mean > 0.7).float()
            probs[:, MATERIAL_TYPES["metal"]] = metal_mask.squeeze()
            
            # Glass classification
            glass_mask = ((metallic_mean < 0.1) & (roughness_mean < 0.2)).float()
            probs[:, MATERIAL_TYPES["glass"]] = glass_mask.squeeze()
            
            # Fabric classification
            fabric_mask = ((metallic_mean < 0.1) & (roughness_mean > 0.7)).float()
            probs[:, MATERIAL_TYPES["fabric"]] = fabric_mask.squeeze()
            
            # Plastic classification (default for non-metal, medium roughness)
            plastic_mask = (
                (metallic_mean < 0.1) & 
                (roughness_mean >= 0.2) & 
                (roughness_mean <= 0.6)
            ).float()
            probs[:, MATERIAL_TYPES["plastic"]] = plastic_mask.squeeze()
            
            # Skin classification (low metallic, medium roughness, warm base color)
            # Check if base color is in warm range (higher R, moderate G)
            base_color_mean = base_color.mean(dim=-1, keepdim=True)  # [N, 1]
            base_color_r = base_color[:, 0:1]  # Red channel
            skin_mask = (
                (metallic_mean < 0.1) & 
                (roughness_mean >= 0.3) & 
                (roughness_mean <= 0.6) &
                (base_color_r > 0.4) & (base_color_r < 0.9)  # Warm skin tones
            ).float()
            probs[:, MATERIAL_TYPES["skin"]] = skin_mask.squeeze()
            
            # Liquid classification (very low roughness, similar to glass but may have color)
            liquid_mask = (
                (metallic_mean < 0.1) & 
                (roughness_mean < 0.15) &
                (roughness_mean >= 0.05)  # Slightly rougher than glass
            ).float()
            probs[:, MATERIAL_TYPES["liquid"]] = liquid_mask.squeeze()
        else:
            # Without metallic, use roughness only
            glass_mask = (roughness_mean < 0.2).float()
            probs[:, MATERIAL_TYPES["glass"]] = glass_mask.squeeze()
            
            fabric_mask = (roughness_mean > 0.7).float()
            probs[:, MATERIAL_TYPES["fabric"]] = fabric_mask.squeeze()
            
            # Default to plastic for medium roughness
            plastic_mask = ((roughness_mean >= 0.2) & (roughness_mean <= 0.6)).float()
            probs[:, MATERIAL_TYPES["plastic"]] = plastic_mask.squeeze()
            
            # Skin classification (medium roughness, warm colors)
            base_color_mean = base_color.mean(dim=-1, keepdim=True)  # [N, 1]
            base_color_r = base_color[:, 0:1]  # Red channel
            skin_mask = (
                (roughness_mean >= 0.3) & 
                (roughness_mean <= 0.6) &
                (base_color_r > 0.4) & (base_color_r < 0.9)
            ).float()
            probs[:, MATERIAL_TYPES["skin"]] = skin_mask.squeeze()
            
            # Liquid classification (very low roughness)
            liquid_mask = ((roughness_mean >= 0.05) & (roughness_mean < 0.15)).float()
            probs[:, MATERIAL_TYPES["liquid"]] = liquid_mask.squeeze()
        
        # Normalize probabilities (soft assignment)
        probs = F.softmax(probs * 10.0, dim=-1)  # Temperature scaling
        
        return probs


class MaterialPriorLoss:
    """Compute material prior constraints loss"""
    
    def __init__(self, use_soft_constraints: bool = True):
        self.use_soft_constraints = use_soft_constraints
    
    def compute_loss(
        self,
        base_color: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        material_probs: torch.Tensor,
        additional_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute material prior constraint loss
        
        Args:
            base_color: [N, 3] base color
            roughness: [N, 1] roughness
            metallic: [N, 1] metallic
            material_probs: [N, num_types] material type probabilities
            additional_params: Additional parameters (anisotropy, specular, etc.)
            
        Returns:
            Scalar loss value
        """
        loss = torch.tensor(0.0, device=base_color.device, dtype=base_color.dtype)
        
        # Iterate over each material type
        for mat_type, mat_idx in MATERIAL_TYPES.items():
            prior = MATERIAL_PRIORS[mat_type]
            probs = material_probs[:, mat_idx:mat_idx+1]  # [N, 1]
            
            # Only apply constraints for points with significant probability
            mask = (probs > 0.1).float()
            
            # Metallic constraint
            if "metallic" in prior:
                target_range = prior["metallic"]
                # Ensure metallic has correct shape [N, 1]
                if metallic.dim() == 1:
                    metallic = metallic.unsqueeze(-1)  # [N] -> [N, 1]
                elif metallic.dim() > 2:
                    metallic = metallic.view(-1, 1)  # Flatten if needed
                
                metallic_clamped = metallic.clamp(target_range[0], target_range[1])
                # probs is [N, 1], mask is [N, 1], metallic is [N, 1]
                mse = (metallic - metallic_clamped) ** 2  # [N, 1]
                loss += (probs * mask * mse).mean()
            
            # Roughness constraint
            if "roughness" in prior:
                target_range = prior["roughness"]
                # Ensure roughness has correct shape [N, 1]
                if roughness.dim() == 1:
                    roughness = roughness.unsqueeze(-1)  # [N] -> [N, 1]
                elif roughness.dim() > 2:
                    roughness = roughness.view(-1, 1)  # Flatten if needed
                
                roughness_clamped = roughness.clamp(target_range[0], target_range[1])
                mse = (roughness - roughness_clamped) ** 2  # [N, 1]
                loss += (probs * mask * mse).mean()
            
            # Additional parameter constraints
            if additional_params is not None:
                for param_name, target_range in prior.items():
                    if param_name in additional_params and param_name not in ["base_color"]:
                        param = additional_params[param_name]
                        if isinstance(target_range, tuple):
                            # Ensure param has correct shape [N, ...]
                            if param.dim() == 1:
                                param = param.unsqueeze(-1)  # [N] -> [N, 1]
                            elif param.dim() > 2:
                                param = param.view(param.shape[0], -1)  # Flatten if needed
                            
                            param_clamped = param.clamp(target_range[0], target_range[1])
                            mse = (param - param_clamped) ** 2  # [N, ...]
                            loss += (probs * mask * mse).mean()
        
        return loss


def get_material_smooth_weight(material_probs: torch.Tensor) -> torch.Tensor:
    """
    Get adaptive smooth weight based on material type
    Metals and glass need stronger smoothing, fabrics allow more variation
    
    Args:
        material_probs: [N, num_types] material probabilities
        
    Returns:
        [N, 1] smooth weights
    """
    # Strong smoothing for metals and glass
    strong_smooth = (
        material_probs[:, MATERIAL_TYPES["metal"]] + 
        material_probs[:, MATERIAL_TYPES["glass"]]
    )
    
    # Medium smoothing for plastic and ceramic
    medium_smooth = (
        material_probs[:, MATERIAL_TYPES["plastic"]] + 
        material_probs[:, MATERIAL_TYPES["ceramic"]]
    )
    
    # Weak smoothing for fabric and wood (allow texture variation)
    weak_smooth = (
        material_probs[:, MATERIAL_TYPES["fabric"]] + 
        material_probs[:, MATERIAL_TYPES["wood"]]
    )
    
    # Medium-strong smoothing for skin and liquid (need moderate smoothing)
    moderate_smooth = (
        material_probs[:, MATERIAL_TYPES["skin"]] + 
        material_probs[:, MATERIAL_TYPES["liquid"]]
    )
    
    weights = strong_smooth * 2.0 + medium_smooth * 1.0 + moderate_smooth * 1.5 + weak_smooth * 0.5
    
    return weights.unsqueeze(-1)  # [N, 1]

