"""
Extended BRDF Models
- Disney/Artist-friendly BRDF
- Full Cook-Torrance microsurface model
- Support for anisotropy, metallic, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict


class CookTorranceBRDF:
    """Full Cook-Torrance BRDF with support for metallic and anisotropic materials"""
    
    @staticmethod
    def fresnel_schlick(
        cos_theta: torch.Tensor,
        F0: torch.Tensor
    ) -> torch.Tensor:
        """
        Fresnel-Schlick approximation
        
        Args:
            cos_theta: Cosine of angle between view/half and normal [..., 1] or [...]
            F0: Base specular reflectance [..., 3]
            
        Returns:
            Fresnel term [..., 3]
        """
        # Ensure cos_theta has correct shape for broadcasting
        if cos_theta.dim() < F0.dim():
            # If cos_theta is missing a dimension, add it
            while cos_theta.dim() < F0.dim():
                cos_theta = cos_theta.unsqueeze(-1)
        elif cos_theta.shape[-1] != 1 and F0.shape[-1] == 3:
            # If cos_theta needs to be expanded for color channels
            if cos_theta.dim() == F0.dim() - 1:
                cos_theta = cos_theta.unsqueeze(-1)
        
        # Compute Fresnel term with proper broadcasting
        one_minus_cos = torch.clamp(1.0 - cos_theta, min=0.0, max=1.0)
        fresnel_factor = torch.pow(one_minus_cos, 5.0)  # [..., 1] or compatible shape
        return F0 + (1.0 - F0) * fresnel_factor
    
    @staticmethod
    def distribution_ggx(
        NdotH: torch.Tensor,
        roughness: torch.Tensor,
        anisotropy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        GGX/Trowbridge-Reitz normal distribution function
        
        Args:
            NdotH: Dot product between normal and half vector [..., 1]
            roughness: Roughness value [..., 1]
            anisotropy: Anisotropy value [..., 1] (optional, 0=isotropic)
            
        Returns:
            Distribution term D [..., 1]
        """
        alpha = roughness * roughness
        alpha2 = alpha * alpha
        
        if anisotropy is not None:
            # Anisotropic GGX
            # TODO: Implement full anisotropic version
            # For now, use isotropic with roughness adjustment
            alpha = alpha * (1.0 + anisotropy.abs() * 0.5)
            alpha2 = alpha * alpha
        
        NdotH_sq = NdotH * NdotH
        denom = (NdotH_sq * (alpha2 - 1.0) + 1.0)
        denom = torch.clamp(denom, min=1e-7)
        D = alpha2 / (np.pi * denom * denom)
        
        return D
    
    @staticmethod
    def geometry_schlick_ggx(
        cos_theta: torch.Tensor,
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """
        Geometry term using Schlick-GGX approximation
        
        Args:
            cos_theta: Cosine of angle (NdotV or NdotL)
            roughness: Roughness value
            
        Returns:
            Geometry term G [..., 1]
        """
        k = (roughness + 1.0) * (roughness + 1.0) / 8.0
        denom = cos_theta * (1.0 - k) + k
        denom = torch.clamp(denom, min=1e-7)
        return cos_theta / denom
    
    @staticmethod
    def geometry_smith(
        NdotV: torch.Tensor,
        NdotL: torch.Tensor,
        roughness: torch.Tensor
    ) -> torch.Tensor:
        """
        Smith geometry term (G = G_V * G_L)
        
        Args:
            NdotV: Dot product between normal and view [..., 1]
            NdotL: Dot product between normal and light [..., 1]
            roughness: Roughness value [..., 1]
            
        Returns:
            Geometry term G [..., 1]
        """
        G_V = CookTorranceBRDF.geometry_schlick_ggx(NdotV, roughness)
        G_L = CookTorranceBRDF.geometry_schlick_ggx(NdotL, roughness)
        return G_V * G_L
    
    @staticmethod
    def cook_torrance_specular(
        normal: torch.Tensor,
        viewdir: torch.Tensor,
        lightdir: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        base_color: torch.Tensor,
        anisotropy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Full Cook-Torrance specular BRDF
        
        Args:
            normal: Surface normal [N, 3]
            viewdir: View direction [N, 3]
            lightdir: Light direction [N, S, 3]
            roughness: Roughness [N, 1]
            metallic: Metallic [N, 1]
            base_color: Base color [N, 3]
            anisotropy: Anisotropy [N, 1] (optional)
            
        Returns:
            Specular BRDF [N, S, 3]
        """
        # Normalize inputs
        normal = F.normalize(normal, dim=-1)  # [N, 3]
        viewdir = F.normalize(viewdir, dim=-1)  # [N, 3]
        lightdir = F.normalize(lightdir, dim=-1)  # [N, S, 3]
        
        # Half vector
        halfdir = F.normalize(viewdir.unsqueeze(1) + lightdir, dim=-1)  # [N, S, 3]
        
        # Dot products
        NdotV = torch.sum(normal.unsqueeze(1) * viewdir.unsqueeze(1), dim=-1, keepdim=True).clamp(min=1e-7, max=1.0)  # [N, S, 1]
        NdotL = torch.sum(normal.unsqueeze(1) * lightdir, dim=-1, keepdim=True).clamp(min=0.0, max=1.0)  # [N, S, 1]
        NdotH = torch.sum(normal.unsqueeze(1) * halfdir, dim=-1, keepdim=True).clamp(min=1e-7, max=1.0)  # [N, S, 1]
        VdotH = torch.sum(viewdir.unsqueeze(1) * halfdir, dim=-1, keepdim=True).clamp(min=1e-7, max=1.0)  # [N, S, 1]
        
        # Distribution term D
        roughness_exp = roughness.unsqueeze(1)  # [N, 1, 1]
        D = CookTorranceBRDF.distribution_ggx(NdotH, roughness_exp, 
                                              anisotropy.unsqueeze(1) if anisotropy is not None else None)  # [N, S, 1]
        
        # Geometry term G
        G = CookTorranceBRDF.geometry_smith(NdotV, NdotL, roughness_exp)  # [N, S, 1]
        
        # Fresnel term F_fresnel (rename to avoid conflict with torch.nn.functional as F)
        F0 = 0.04 * (1.0 - metallic.unsqueeze(1)) + base_color.unsqueeze(1) * metallic.unsqueeze(1)  # [N, 1, 3]
        # VdotH is [N, S, 1], F0 is [N, 1, 3], broadcasting will work correctly
        F_fresnel = CookTorranceBRDF.fresnel_schlick(VdotH, F0)  # [N, S, 3]
        
        # Cook-Torrance specular BRDF
        denominator = 4.0 * NdotV * NdotL
        denominator = torch.clamp(denominator, min=1e-7)
        
        specular = (D * G * F_fresnel) / denominator  # [N, S, 3]
        
        return specular
    
    @staticmethod
    def lambert_diffuse(
        base_color: torch.Tensor,
        metallic: torch.Tensor
    ) -> torch.Tensor:
        """
        Lambertian diffuse BRDF
        
        Args:
            base_color: Base color [N, 3]
            metallic: Metallic [N, 1] or [N]
            
        Returns:
            Diffuse BRDF coefficient [N, 3]
        """
        # Ensure metallic has correct shape
        if metallic.dim() == 1:
            metallic = metallic.unsqueeze(-1)  # [N] -> [N, 1]
        elif metallic.dim() == 0:
            metallic = metallic.unsqueeze(0).unsqueeze(-1)  # scalar -> [1, 1]
        
        # Ensure same device and dtype
        metallic = metallic.to(base_color.device, dtype=base_color.dtype)
        
        # Non-metallic materials have diffuse component
        # (1.0 - metallic) is [N, 1], base_color is [N, 3], broadcasting works
        diffuse = (1.0 - metallic) * base_color / np.pi
        return diffuse


class DisneyBRDF:
    """Disney/Artist-friendly BRDF"""
    
    @staticmethod
    def subsurface(
        base_color: torch.Tensor,
        subsurface: torch.Tensor,
        NdotL: torch.Tensor,
        NdotV: torch.Tensor
    ) -> torch.Tensor:
        """Subsurface scattering term"""
        # Simplified subsurface scattering
        # Full implementation would include directional albedo
        factor = 1.25 * (1.0 - torch.pow(1.0 - NdotL, 5.0)) * (1.0 - torch.pow(1.0 - NdotV, 5.0))
        return base_color.unsqueeze(1) * subsurface.unsqueeze(1) * factor
    
    @staticmethod
    def metallic(
        base_color: torch.Tensor,
        metallic: torch.Tensor
    ) -> torch.Tensor:
        """Metallic term"""
        return metallic.unsqueeze(1)
    
    @staticmethod
    def specular(
        specular: torch.Tensor,
        NdotV: torch.Tensor
    ) -> torch.Tensor:
        """Specular term"""
        return specular.unsqueeze(1) * (0.04 + 0.96 * torch.pow(1.0 - NdotV, 5.0))
    
    @staticmethod
    def clearcoat(
        clearcoat: torch.Tensor,
        clearcoat_roughness: torch.Tensor,
        NdotH: torch.Tensor,
        NdotV: torch.Tensor,
        NdotL: torch.Tensor
    ) -> torch.Tensor:
        """Clearcoat term (for materials like car paint)"""
        # Simplified clearcoat using Cook-Torrance
        roughness = clearcoat_roughness.unsqueeze(1)
        D = CookTorranceBRDF.distribution_ggx(NdotH, roughness)
        G = CookTorranceBRDF.geometry_smith(NdotV, NdotL, roughness)
        F_fresnel = 0.04 + 0.96 * torch.pow(1.0 - torch.sum(NdotV, dim=-1, keepdim=True), 5.0)
        
        denominator = 4.0 * NdotV * NdotL
        denominator = torch.clamp(denominator, min=1e-7)
        
        return clearcoat.unsqueeze(1) * (D * G * F_fresnel) / denominator
    
    @staticmethod
    def sheen(
        sheen: torch.Tensor,
        sheen_tint: torch.Tensor,
        base_color: torch.Tensor,
        NdotV: torch.Tensor
    ) -> torch.Tensor:
        """Sheen term (for materials like velvet)"""
        factor = torch.pow(1.0 - NdotV, 5.0)
        return sheen.unsqueeze(1) * (sheen_tint.unsqueeze(1) * base_color.unsqueeze(1) + (1.0 - sheen_tint.unsqueeze(1))) * factor
    
    @staticmethod
    def disney_brdf(
        normal: torch.Tensor,
        viewdir: torch.Tensor,
        lightdir: torch.Tensor,
        base_color: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        specular: Optional[torch.Tensor] = None,
        subsurface: Optional[torch.Tensor] = None,
        clearcoat: Optional[torch.Tensor] = None,
        clearcoat_roughness: Optional[torch.Tensor] = None,
        sheen: Optional[torch.Tensor] = None,
        sheen_tint: Optional[torch.Tensor] = None,
        anisotropy: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full Disney BRDF
        
        Returns:
            Dictionary with diffuse, specular, and other terms
        """
        # Normalize
        normal = F.normalize(normal, dim=-1)
        viewdir = F.normalize(viewdir, dim=-1)
        lightdir = F.normalize(lightdir, dim=-1)
        
        # Dot products
        NdotV = torch.sum(normal.unsqueeze(1) * viewdir.unsqueeze(1), dim=-1, keepdim=True).clamp(min=1e-7, max=1.0)
        NdotL = torch.sum(normal.unsqueeze(1) * lightdir, dim=-1, keepdim=True).clamp(min=0.0, max=1.0)
        halfdir = F.normalize(viewdir.unsqueeze(1) + lightdir, dim=-1)
        NdotH = torch.sum(normal.unsqueeze(1) * halfdir, dim=-1, keepdim=True).clamp(min=1e-7, max=1.0)
        
        # Base diffuse
        diffuse = CookTorranceBRDF.lambert_diffuse(base_color, metallic).unsqueeze(1)  # [N, S, 3]
        
        # Subsurface
        if subsurface is not None:
            subsurface_term = DisneyBRDF.subsurface(base_color, subsurface, NdotL, NdotV)
            diffuse = diffuse + subsurface_term
        
        # Specular (using Cook-Torrance)
        specular_term = CookTorranceBRDF.cook_torrance_specular(
            normal, viewdir.squeeze(1), lightdir.squeeze(1),
            roughness, metallic, base_color, anisotropy
        )
        
        # Clearcoat
        clearcoat_term = None
        if clearcoat is not None and clearcoat_roughness is not None:
            clearcoat_term = DisneyBRDF.clearcoat(clearcoat, clearcoat_roughness, NdotH, NdotV, NdotL)
        
        # Sheen
        sheen_term = None
        if sheen is not None:
            if sheen_tint is None:
                sheen_tint = torch.zeros_like(sheen)
            sheen_term = DisneyBRDF.sheen(sheen, sheen_tint, base_color, NdotV)
        
        return {
            "diffuse": diffuse,
            "specular": specular_term,
            "clearcoat": clearcoat_term,
            "sheen": sheen_term
        }


def select_brdf_model(
    model_type: str = "cook_torrance"
) -> type:
    """
    Select BRDF model
    
    Args:
        model_type: "cook_torrance" or "disney"
        
    Returns:
        BRDF model class
    """
    if model_type == "disney":
        return DisneyBRDF
    else:
        return CookTorranceBRDF

