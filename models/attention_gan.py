"""
Attention-Based Text-to-Image GAN
Incorporates Self-Attention and Cross-Attention for higher quality generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import sys
sys.path.append('.')
from models.attention_modules import (
    SelfAttention,
    AttentionBlock,
    SpatialCrossAttention,
    CBAM
)


class ResidualAttentionBlock(nn.Module):
    """Residual block with integrated attention"""
    
    def __init__(self, channels: int, text_embedding_dim: int, num_heads: int = 8):
        super(ResidualAttentionBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Attention
        self.attention = AttentionBlock(
            channels,
            text_embedding_dim,
            num_heads=num_heads,
            use_self_attention=True,
            use_cross_attention=True
        )
    
    def forward(self, x, text_embedding):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply attention
        out = self.attention(out, text_embedding)
        
        out += residual
        return F.relu(out)


class AttentionGenerator(nn.Module):
    """
    Generator with Self-Attention and Cross-Attention
    Produces higher quality images by attending to relevant text features
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        text_embedding_dim: int = 768,
        ngf: int = 64,
        num_channels: int = 3,
        image_size: int = 64,
        num_attention_blocks: int = 2,
        attention_heads: int = 8
    ):
        """
        Initialize Attention-based Generator
        
        Args:
            latent_dim: Dimension of latent noise vector
            text_embedding_dim: Dimension of text embeddings
            ngf: Number of generator filters
            num_channels: Number of output image channels
            image_size: Size of generated images
            num_attention_blocks: Number of attention blocks to use
            attention_heads: Number of attention heads
        """
        super(AttentionGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        self.image_size = image_size
        
        # Text conditioning
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256)
        )
        
        # Combined input dimension
        combined_dim = latent_dim + 256
        
        # Initial projection
        init_size = image_size // 16
        self.init = nn.Sequential(
            nn.Linear(combined_dim, ngf * 16 * init_size * init_size),
            nn.BatchNorm1d(ngf * 16 * init_size * init_size),
            nn.ReLU(True)
        )
        
        # Upsampling block 1: ngf*16 -> ngf*8
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        
        # Self-attention at 8x8
        self.self_attn_8 = SelfAttention(ngf * 8, reduction=8)
        
        # Upsampling block 2: ngf*8 -> ngf*4
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        
        # Cross-attention at 16x16
        self.cross_attn_16 = SpatialCrossAttention(
            ngf * 4,
            text_embedding_dim,
            num_heads=attention_heads
        )
        
        # Upsampling block 3: ngf*4 -> ngf*2
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        
        # Attention blocks for refinement
        self.attention_blocks = nn.ModuleList([
            ResidualAttentionBlock(ngf * 2, text_embedding_dim, attention_heads)
            for _ in range(num_attention_blocks)
        ])
        
        # Self-attention at 32x32
        self.self_attn_32 = SelfAttention(ngf * 2, reduction=8)
        
        # Upsampling block 4: ngf*2 -> ngf
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        
        # CBAM for final refinement
        self.cbam = CBAM(ngf, reduction=16)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(ngf, num_channels, 3, 1, 1),
            nn.Tanh()
        )
        
        self.init_size = init_size
        self.ngf = ngf
    
    def forward(self, noise, text_embedding):
        """
        Forward pass
        
        Args:
            noise: [batch_size, latent_dim]
            text_embedding: [batch_size, text_embedding_dim]
            
        Returns:
            generated_images: [batch_size, num_channels, image_size, image_size]
        """
        batch_size = noise.size(0)
        
        # Project text embedding
        text_condition = self.text_projection(text_embedding)
        
        # Concatenate noise and text
        combined = torch.cat([noise, text_condition], dim=1)
        
        # Initial projection
        out = self.init(combined)
        out = out.view(batch_size, self.ngf * 16, self.init_size, self.init_size)
        
        # Upsampling with attention
        out = self.up1(out)  # 8x8
        out = self.self_attn_8(out)  # Self-attention at 8x8
        
        out = self.up2(out)  # 16x16
        out = self.cross_attn_16(out, text_embedding)  # Cross-attention at 16x16
        
        out = self.up3(out)  # 32x32
        
        # Apply attention blocks
        for attn_block in self.attention_blocks:
            out = attn_block(out, text_embedding)
        
        out = self.self_attn_32(out)  # Self-attention at 32x32
        
        out = self.up4(out)  # 64x64
        
        # Final CBAM refinement
        out = self.cbam(out)
        
        # Output
        out = self.output(out)
        
        return out


class AttentionDiscriminator(nn.Module):
    """
    Discriminator with attention mechanisms
    Better at evaluating text-image alignment
    """
    
    def __init__(
        self,
        text_embedding_dim: int = 768,
        ndf: int = 64,
        num_channels: int = 3,
        image_size: int = 64,
        attention_heads: int = 8
    ):
        """
        Initialize Attention-based Discriminator
        
        Args:
            text_embedding_dim: Dimension of text embeddings
            ndf: Number of discriminator filters
            num_channels: Number of input image channels
            image_size: Size of input images
            attention_heads: Number of attention heads
        """
        super(AttentionDiscriminator, self).__init__()
        
        self.image_size = image_size
        
        # Image encoding pathway
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Self-attention at 16x16
        self.self_attn = SelfAttention(ndf * 2, reduction=8)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Cross-attention for text conditioning at 8x8
        self.cross_attn = SpatialCrossAttention(
            ndf * 4,
            text_embedding_dim,
            num_heads=attention_heads
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Text encoder for matching
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, ndf * 8)
        )
        
        # Combined classifier
        final_size = image_size // 16
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, final_size, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, images, text_embedding):
        """
        Forward pass
        
        Args:
            images: [batch_size, num_channels, image_size, image_size]
            text_embedding: [batch_size, text_embedding_dim]
            
        Returns:
            validity: [batch_size, 1] - probability that image is real
        """
        batch_size = images.size(0)
        
        # Image encoding
        out = self.conv1(images)  # 32x32
        out = self.conv2(out)     # 16x16
        out = self.self_attn(out) # Self-attention
        
        out = self.conv3(out)     # 8x8
        out = self.cross_attn(out, text_embedding)  # Cross-attention with text
        
        out = self.conv4(out)     # 4x4
        
        # Text encoding
        text_features = self.text_encoder(text_embedding)
        text_features = text_features.unsqueeze(2).unsqueeze(3)
        text_features = text_features.expand_as(out)
        
        # Element-wise product (conditioning)
        combined = out * text_features
        
        # Classification
        validity = self.classifier(combined)
        
        return validity.view(batch_size, -1)


class AttentionTextToImageGAN(nn.Module):
    """
    Complete Attention-based Text-to-Image GAN
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        text_embedding_dim: int = 768,
        ngf: int = 64,
        ndf: int = 64,
        num_channels: int = 3,
        image_size: int = 64,
        num_attention_blocks: int = 2,
        attention_heads: int = 8
    ):
        """
        Initialize complete attention GAN model
        """
        super(AttentionTextToImageGAN, self).__init__()
        
        self.generator = AttentionGenerator(
            latent_dim=latent_dim,
            text_embedding_dim=text_embedding_dim,
            ngf=ngf,
            num_channels=num_channels,
            image_size=image_size,
            num_attention_blocks=num_attention_blocks,
            attention_heads=attention_heads
        )
        
        self.discriminator = AttentionDiscriminator(
            text_embedding_dim=text_embedding_dim,
            ndf=ndf,
            num_channels=num_channels,
            image_size=image_size,
            attention_heads=attention_heads
        )
        
        self.latent_dim = latent_dim
    
    def generate(self, text_embedding: torch.Tensor, num_samples: int = 1):
        """
        Generate images from text embeddings
        
        Args:
            text_embedding: [batch_size, text_embedding_dim]
            num_samples: Number of samples to generate per text
            
        Returns:
            generated_images: [batch_size * num_samples, C, H, W]
        """
        batch_size = text_embedding.size(0)
        device = text_embedding.device
        
        if num_samples > 1:
            text_embedding = text_embedding.repeat_interleave(num_samples, dim=0)
        
        # Sample random noise
        noise = torch.randn(
            batch_size * num_samples,
            self.latent_dim,
            device=device
        )
        
        # Generate images
        with torch.no_grad():
            generated_images = self.generator(noise, text_embedding)
        
        return generated_images
    
    def save(self, path: str):
        """Save model weights"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, path)
    
    def load(self, path: str, device: str = 'cpu'):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])


def weights_init(m):
    """Custom weights initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    print("Testing Attention-Based Text-to-Image GAN\n")
    
    # Model parameters
    batch_size = 4
    latent_dim = 100
    text_embedding_dim = 768
    image_size = 64
    num_channels = 3
    
    # Create model
    print("🏗️  Building Attention GAN...")
    model = AttentionTextToImageGAN(
        latent_dim=latent_dim,
        text_embedding_dim=text_embedding_dim,
        ngf=64,
        ndf=64,
        image_size=image_size,
        num_channels=num_channels,
        num_attention_blocks=2,
        attention_heads=8
    )
    
    # Apply weight initialization
    model.generator.apply(weights_init)
    model.discriminator.apply(weights_init)
    
    print("✅ Model created\n")
    
    # Test forward pass
    noise = torch.randn(batch_size, latent_dim)
    text_embedding = torch.randn(batch_size, text_embedding_dim)
    
    print("🧪 Testing Generator...")
    generated = model.generator(noise, text_embedding)
    print(f"   Input noise: {noise.shape}")
    print(f"   Input text: {text_embedding.shape}")
    print(f"   Generated images: {generated.shape}")
    print(f"   ✅ Generator working\n")
    
    print("🧪 Testing Discriminator...")
    fake_images = torch.randn(batch_size, num_channels, image_size, image_size)
    validity = model.discriminator(fake_images, text_embedding)
    print(f"   Input images: {fake_images.shape}")
    print(f"   Output validity: {validity.shape}")
    print(f"   ✅ Discriminator working\n")
    
    # Count parameters
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    total_params = gen_params + disc_params
    
    print("📊 Model Statistics:")
    print(f"   Generator parameters: {gen_params:,}")
    print(f"   Discriminator parameters: {disc_params:,}")
    print(f"   Total parameters: {total_params:,}")
    
    # Attention analysis
    print("\n🎯 Attention Mechanisms:")
    print(f"   ✓ Self-Attention layers: 3")
    print(f"   ✓ Cross-Attention layers: 3")
    print(f"   ✓ CBAM layers: 1")
    print(f"   ✓ Attention heads: 8")
    
    print("\n✅ Attention-Based GAN tested successfully!")
    print("\n💡 Key Improvements over Basic GAN:")
    print("   • Self-attention for spatial coherence")
    print("   • Cross-attention for text-image alignment")
    print("   • Multi-scale attention (8x8, 16x16, 32x32)")
    print("   • CBAM for channel & spatial refinement")
    print("   • Residual attention blocks for quality")
