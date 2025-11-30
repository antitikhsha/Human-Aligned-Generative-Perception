"""
StyleGAN2-ADA with HPE Integration
Modified StyleGAN2-ADA architecture that incorporates Human Perceptual Embeddings
in the discriminator for human-aligned generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
import logging

from ..hpe.hpe_core import HPEEmbedder, compute_hpe_similarity

logger = logging.getLogger(__name__)


class ModulatedConv2d(nn.Module):
    """Modulated convolution layer for StyleGAN2"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        upsample: bool = False,
        demodulate: bool = True,
        blur_kernel: List[int] = [1, 3, 3, 1]
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        
        # Weight modulation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Style modulation
        self.modulation = nn.Linear(style_dim, in_channels, bias=True)
        nn.init.constant_(self.modulation.weight, 1.0)
        nn.init.constant_(self.modulation.bias, 0.0)
        
        # Blur kernel for upsampling
        if upsample:
            factor = 2
            p = len(blur_kernel) - factor
            self.register_buffer('blur_kernel', torch.tensor(blur_kernel, dtype=torch.float32))
            self.pad = ((p + 1) // 2, p // 2)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Modulate weights
        style = self.modulation(style).view(batch_size, 1, self.in_channels, 1, 1)
        weight = self.weight.unsqueeze(0) * style
        
        # Demodulation
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch_size, self.out_channels, 1, 1, 1)
        
        # Reshape for grouped convolution
        weight = weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        x = x.view(1, -1, x.shape[2], x.shape[3])
        
        # Convolution
        if self.upsample:
            x = F.conv_transpose2d(x, weight, padding=0, groups=batch_size)
            x = self.blur(x)
        else:
            x = F.conv2d(x, weight, padding=self.kernel_size//2, groups=batch_size)
        
        # Reshape back and add bias
        x = x.view(batch_size, self.out_channels, x.shape[2], x.shape[3])
        x = x + self.bias.view(1, -1, 1, 1)
        
        return x
    
    def blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply blur kernel"""
        if not hasattr(self, 'blur_kernel'):
            return x
        
        kernel = self.blur_kernel[None, None, :].repeat(x.shape[1], 1, 1)
        kernel = kernel.view(-1, 1, len(self.blur_kernel))
        
        x = F.conv1d(x.view(-1, x.shape[2], x.shape[3]), kernel, padding=self.pad, groups=x.shape[1])
        x = x.view(x.shape[0], -1, x.shape[2])
        
        kernel = self.blur_kernel[None, :, None].repeat(x.shape[1], 1, 1)
        kernel = kernel.view(-1, len(self.blur_kernel), 1)
        
        x = F.conv1d(x, kernel, padding=self.pad, groups=x.shape[1])
        
        return x.view(x.shape[0], -1, x.shape[2], x.shape[3])


class MappingNetwork(nn.Module):
    """StyleGAN2 mapping network Z -> W"""
    
    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        num_layers: int = 8,
        lr_mul: float = 0.01
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.extend([
                nn.Linear(in_dim, w_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        self.mapping = nn.Sequential(*layers)
        
        # Apply learning rate multiplier
        for layer in self.mapping:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= lr_mul
                layer.bias.data *= lr_mul
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent code z to intermediate latent w"""
        return self.mapping(z)


class SynthesisBlock(nn.Module):
    """StyleGAN2 synthesis block"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        resolution: int,
        is_first: bool = False,
        use_noise: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_first = is_first
        self.use_noise = use_noise
        
        if is_first:
            # First block starts with learned constant
            self.const = nn.Parameter(torch.ones(1, in_channels, 4, 4))
            self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, style_dim)
        else:
            # Upsample and convolve
            self.conv0 = ModulatedConv2d(in_channels, out_channels, 3, style_dim, upsample=True)
            self.conv1 = ModulatedConv2d(out_channels, out_channels, 3, style_dim)
        
        # Noise inputs
        if use_noise:
            self.noise_scale0 = nn.Parameter(torch.zeros(1))
            self.noise_scale1 = nn.Parameter(torch.zeros(1))
            self.register_buffer('noise0', torch.randn(1, 1, resolution, resolution))
            self.register_buffer('noise1', torch.randn(1, 1, resolution, resolution))
        
        # Activation
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(
        self, 
        x: torch.Tensor, 
        style0: torch.Tensor, 
        style1: torch.Tensor,
        noise0: Optional[torch.Tensor] = None,
        noise1: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        if self.is_first:
            # Start with learned constant
            batch_size = style0.shape[0]
            x = self.const.repeat(batch_size, 1, 1, 1)
            
            # First convolution
            x = self.conv1(x, style0)
            if self.use_noise:
                if noise0 is None:
                    noise0 = self.noise0
                x = x + self.noise_scale0 * noise0
            x = self.activation(x)
        else:
            # Upsample and convolve
            x = self.conv0(x, style0)
            if self.use_noise:
                if noise0 is None:
                    noise0 = self.noise0
                x = x + self.noise_scale0 * noise0
            x = self.activation(x)
            
            # Second convolution
            x = self.conv1(x, style1)
            if self.use_noise:
                if noise1 is None:
                    noise1 = self.noise1
                x = x + self.noise_scale1 * noise1
            x = self.activation(x)
        
        return x


class Generator(nn.Module):
    """StyleGAN2-ADA Generator with HPE conditioning"""
    
    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        img_channels: int = 3,
        img_resolution: int = 256,
        mapping_layers: int = 8,
        hpe_dim: int = 66,
        use_hpe_conditioning: bool = True
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_channels = img_channels
        self.img_resolution = img_resolution
        self.use_hpe_conditioning = use_hpe_conditioning
        
        # HPE conditioning
        if use_hpe_conditioning:
            self.hpe_projection = nn.Linear(hpe_dim, w_dim)
            self.style_fusion = nn.Linear(w_dim * 2, w_dim)
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim, mapping_layers)
        
        # Synthesis network
        self.synthesis_blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()
        
        # Define channel progression
        channels = [512, 512, 512, 512, 256, 128, 64, 32]
        resolutions = [4, 8, 16, 32, 64, 128, 256, 512]
        
        # Filter to target resolution
        target_idx = int(np.log2(img_resolution)) - 2
        channels = channels[:target_idx + 1]
        resolutions = resolutions[:target_idx + 1]
        
        for i, (res, ch) in enumerate(zip(resolutions, channels)):
            is_first = (i == 0)
            in_ch = channels[i-1] if not is_first else 512
            
            # Synthesis block
            block = SynthesisBlock(
                in_channels=in_ch if not is_first else 512,
                out_channels=ch,
                style_dim=w_dim,
                resolution=res,
                is_first=is_first
            )
            self.synthesis_blocks.append(block)
            
            # RGB output layer
            to_rgb = ModulatedConv2d(ch, img_channels, 1, w_dim, demodulate=False)
            self.to_rgb_layers.append(to_rgb)
        
        self.num_layers = len(self.synthesis_blocks)
    
    def forward(
        self,
        z: torch.Tensor,
        hpe_condition: Optional[torch.Tensor] = None,
        truncation_psi: float = 1.0,
        noise_mode: str = 'random'
    ) -> torch.Tensor:
        """
        Generate images
        
        Args:
            z: Latent codes [batch_size, z_dim]
            hpe_condition: HPE conditioning vectors [batch_size, hpe_dim]
            truncation_psi: Truncation strength
            noise_mode: Noise mode ('random', 'const', or 'none')
            
        Returns:
            Generated images [batch_size, channels, height, width]
        """
        batch_size = z.shape[0]
        
        # Map to intermediate latent space
        w = self.mapping(z)
        
        # HPE conditioning
        if self.use_hpe_conditioning and hpe_condition is not None:
            hpe_w = self.hpe_projection(hpe_condition)
            w_combined = torch.cat([w, hpe_w], dim=1)
            w = self.style_fusion(w_combined)
            w = F.leaky_relu(w, 0.2)
        
        # Apply truncation
        if truncation_psi < 1.0:
            w_avg = w.mean(dim=0, keepdim=True)
            w = w_avg + truncation_psi * (w - w_avg)
        
        # Prepare styles (broadcast w for all layers)
        styles = w.unsqueeze(1).repeat(1, self.num_layers * 2, 1)
        
        # Synthesis
        x = None
        rgb_out = None
        
        for i, (block, to_rgb) in enumerate(zip(self.synthesis_blocks, self.to_rgb_layers)):
            # Get styles for this block
            style0 = styles[:, i * 2]
            style1 = styles[:, i * 2 + 1] if i > 0 else style0
            
            # Generate noise
            if noise_mode == 'random':
                noise0 = torch.randn_like(x[:, :1]) if x is not None else None
                noise1 = torch.randn_like(x[:, :1]) if x is not None else None
            else:
                noise0 = noise1 = None
            
            # Forward through block
            x = block(x, style0, style1, noise0, noise1)
            
            # RGB output
            rgb = to_rgb(x, style0)
            if rgb_out is not None:
                rgb_out = F.interpolate(rgb_out, scale_factor=2, mode='bilinear', align_corners=False)
                rgb_out = rgb_out + rgb
            else:
                rgb_out = rgb
        
        return rgb_out


class HPEDiscriminator(nn.Module):
    """
    StyleGAN2-ADA Discriminator enhanced with Human Perceptual Embeddings
    """
    
    def __init__(
        self,
        img_channels: int = 3,
        img_resolution: int = 256,
        hpe_dim: int = 66,
        use_hpe: bool = True
    ):
        super().__init__()
        
        self.img_channels = img_channels
        self.img_resolution = img_resolution
        self.use_hpe = use_hpe
        
        # Build discriminator blocks
        self.disc_blocks = nn.ModuleList()
        self.from_rgb_layers = nn.ModuleList()
        
        # Channel progression (reverse of generator)
        channels = [32, 64, 128, 256, 512, 512, 512, 512]
        resolutions = [512, 256, 128, 64, 32, 16, 8, 4]
        
        # Filter to target resolution
        target_idx = int(np.log2(img_resolution)) - 2
        channels = channels[-target_idx-1:]
        resolutions = resolutions[-target_idx-1:]
        
        for i, (res, ch) in enumerate(zip(resolutions, channels)):
            # From RGB layer
            from_rgb = nn.Conv2d(img_channels if i == 0 else channels[i-1], ch, 1)
            self.from_rgb_layers.append(from_rgb)
            
            # Discriminator block
            next_ch = channels[i+1] if i < len(channels)-1 else 512
            block = self._make_disc_block(ch, next_ch)
            self.disc_blocks.append(block)
        
        # Final layers
        self.final_conv = nn.Conv2d(512, 512, 3, padding=1)
        self.final_linear = nn.Linear(512 * 4 * 4, 512)
        
        # Standard adversarial output
        self.adv_head = nn.Linear(512, 1)
        
        # HPE integration
        if use_hpe:
            self.hpe_embedder = HPEEmbedder(
                input_dim=512,  # Features from discriminator backbone
                embed_dim=hpe_dim,
                hidden_dims=[512, 256]
            )
            
            # HPE-aware adversarial head
            self.hpe_adv_head = nn.Linear(512 + hpe_dim, 1)
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def _make_disc_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a discriminator block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2)
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        hpe_target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HPE-enhanced discriminator
        
        Args:
            x: Input images [batch_size, channels, height, width]
            hpe_target: Target HPE embeddings for conditioning [batch_size, hpe_dim]
            
        Returns:
            Dictionary containing discriminator outputs
        """
        # Forward through discriminator blocks
        rgb_in = x
        
        for i, (from_rgb, block) in enumerate(zip(self.from_rgb_layers, self.disc_blocks)):
            if i == 0:
                x = from_rgb(rgb_in)
            else:
                # Skip connection from RGB
                rgb_in = F.avg_pool2d(rgb_in, 2)
                x = x + from_rgb(rgb_in)
            
            x = block(x)
        
        # Final processing
        x = self.final_conv(x)
        x = self.activation(x)
        
        # Global average pooling
        features = F.adaptive_avg_pool2d(x, 1).flatten(1)
        features = self.final_linear(features)
        features = self.activation(features)
        
        outputs = {}
        
        # Standard adversarial loss
        outputs['adv_logits'] = self.adv_head(features)
        
        # HPE integration
        if self.use_hpe:
            # Generate HPE embeddings from image features
            hpe_pred = self.hpe_embedder(features)
            outputs['hpe_embeddings'] = hpe_pred
            
            # HPE-conditioned adversarial loss
            if hpe_target is not None:
                hpe_features = torch.cat([features, hpe_target], dim=1)
                outputs['hpe_adv_logits'] = self.hpe_adv_head(hpe_features)
            
            # Combined features for loss computation
            combined_features = torch.cat([features, hpe_pred], dim=1)
            outputs['hpe_combined_logits'] = self.hpe_adv_head(combined_features)
        
        outputs['features'] = features
        
        return outputs


class HPEStyleGAN2(nn.Module):
    """
    Complete HPE-StyleGAN2 model combining generator and discriminator
    """
    
    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        img_channels: int = 3,
        img_resolution: int = 256,
        hpe_dim: int = 66
    ):
        super().__init__()
        
        self.generator = Generator(
            z_dim=z_dim,
            w_dim=w_dim,
            img_channels=img_channels,
            img_resolution=img_resolution,
            hpe_dim=hpe_dim,
            use_hpe_conditioning=True
        )
        
        self.discriminator = HPEDiscriminator(
            img_channels=img_channels,
            img_resolution=img_resolution,
            hpe_dim=hpe_dim,
            use_hpe=True
        )
    
    def forward(self, z: torch.Tensor, hpe_condition: Optional[torch.Tensor] = None):
        """Forward pass for inference"""
        return self.generator(z, hpe_condition)


if __name__ == "__main__":
    # Test the model
    print("Testing HPE-StyleGAN2 Architecture...")
    
    # Initialize model
    model = HPEStyleGAN2(
        z_dim=512,
        w_dim=512,
        img_channels=3,
        img_resolution=256,
        hpe_dim=66
    )
    
    # Test generator
    z = torch.randn(2, 512)
    hpe_condition = torch.randn(2, 66)
    
    fake_images = model.generator(z, hpe_condition)
    print(f"Generated images shape: {fake_images.shape}")
    
    # Test discriminator
    disc_output = model.discriminator(fake_images, hpe_condition)
    print(f"Discriminator outputs: {list(disc_output.keys())}")
    
    # Count parameters
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    print(f"Total parameters: {gen_params + disc_params:,}")
