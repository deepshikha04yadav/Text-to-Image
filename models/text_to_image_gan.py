"""
GAN-based Text-to-Image Generator
Implements conditional GAN architecture for text-to-image synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class TextConditioningModule(nn.Module):
    """
    Conditions the generator on text embeddings
    Uses projection and modulation techniques
    """
    
    def __init__(self, text_embedding_dim: int, condition_dim: int):
        super(TextConditioningModule, self).__init__()
        
        # Project text embeddings to conditioning vector
        self.projection = nn.Sequential(
            nn.Linear(text_embedding_dim, condition_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(condition_dim * 2, condition_dim),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, text_embedding):
        """
        Args:
            text_embedding: [batch_size, text_embedding_dim]
        Returns:
            condition: [batch_size, condition_dim]
        """
        return self.projection(text_embedding)


class Generator(nn.Module):
    """
    Text-conditioned GAN Generator
    Generates images from random noise and text embeddings
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        text_embedding_dim: int = 768,
        ngf: int = 64,
        num_channels: int = 3,
        image_size: int = 64,
        num_residual_blocks: int = 3
    ):
        """
        Initialize Generator
        
        Args:
            latent_dim: Dimension of latent noise vector
            text_embedding_dim: Dimension of text embeddings (768 for BERT/CLIP)
            ngf: Number of generator filters in first conv layer
            num_channels: Number of output image channels (3 for RGB)
            image_size: Size of generated images
            num_residual_blocks: Number of residual blocks
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        self.image_size = image_size
        
        # Text conditioning module
        self.text_conditioner = TextConditioningModule(
            text_embedding_dim, 
            condition_dim=256
        )
        
        # Combined input dimension (noise + text condition)
        combined_dim = latent_dim + 256
        
        # Initial projection
        init_size = image_size // 16  # Will upscale 4 times (2^4 = 16)
        self.init = nn.Sequential(
            nn.Linear(combined_dim, ngf * 16 * init_size * init_size),
            nn.BatchNorm1d(ngf * 16 * init_size * init_size),
            nn.ReLU(True)
        )
        
        # Upsampling blocks
        self.conv_blocks = nn.ModuleList()
        
        # Block 1: ngf*16 -> ngf*8
        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True)
            )
        )
        
        # Block 2: ngf*8 -> ngf*4
        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True)
            )
        )
        
        # Block 3: ngf*4 -> ngf*2
        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True)
            )
        )
        
        # Residual blocks for refinement
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(ngf * 2) for _ in range(num_residual_blocks)
        ])
        
        # Block 4: ngf*2 -> ngf
        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True)
            )
        )
        
        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(ngf, num_channels, 3, 1, 1),
            nn.Tanh()  # Output in range [-1, 1]
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
        
        # Condition on text
        text_condition = self.text_conditioner(text_embedding)
        
        # Concatenate noise and text condition
        combined = torch.cat([noise, text_condition], dim=1)
        
        # Initial projection
        out = self.init(combined)
        out = out.view(batch_size, self.ngf * 16, self.init_size, self.init_size)
        
        # Upsampling
        for conv_block in self.conv_blocks[:3]:
            out = conv_block(out)
        
        # Residual refinement
        for residual_block in self.residual_blocks:
            out = residual_block(out)
        
        # Final upsampling
        out = self.conv_blocks[3](out)
        
        # Output
        out = self.output(out)
        
        return out


class Discriminator(nn.Module):
    """
    Text-conditioned GAN Discriminator
    Determines if image-text pairs are real or fake
    """
    
    def __init__(
        self,
        text_embedding_dim: int = 768,
        ndf: int = 64,
        num_channels: int = 3,
        image_size: int = 64
    ):
        """
        Initialize Discriminator
        
        Args:
            text_embedding_dim: Dimension of text embeddings
            ndf: Number of discriminator filters
            num_channels: Number of input image channels
            image_size: Size of input images
        """
        super(Discriminator, self).__init__()
        
        self.image_size = image_size
        
        # Image encoding pathway
        self.image_encoder = nn.Sequential(
            # input: [batch, num_channels, image_size, image_size]
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state: [batch, ndf, image_size/2, image_size/2]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state: [batch, ndf*2, image_size/4, image_size/4]
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state: [batch, ndf*4, image_size/8, image_size/8]
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Text conditioning
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, ndf * 8)
        )
        
        # Combined discriminator
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
        
        # Encode image
        img_features = self.image_encoder(images)
        # [batch_size, ndf*8, image_size/16, image_size/16]
        
        # Encode text
        text_features = self.text_encoder(text_embedding)
        # [batch_size, ndf*8]
        
        # Spatially replicate text features
        text_features = text_features.unsqueeze(2).unsqueeze(3)
        text_features = text_features.expand_as(img_features)
        
        # Element-wise product (conditioning)
        combined = img_features * text_features
        
        # Classification
        validity = self.classifier(combined)
        
        return validity.view(batch_size, -1)


class TextToImageGAN(nn.Module):
    """
    Complete Text-to-Image GAN model
    Combines Generator and Discriminator
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        text_embedding_dim: int = 768,
        ngf: int = 64,
        ndf: int = 64,
        num_channels: int = 3,
        image_size: int = 64,
        num_residual_blocks: int = 3
    ):
        """
        Initialize complete GAN model
        """
        super(TextToImageGAN, self).__init__()
        
        self.generator = Generator(
            latent_dim=latent_dim,
            text_embedding_dim=text_embedding_dim,
            ngf=ngf,
            num_channels=num_channels,
            image_size=image_size,
            num_residual_blocks=num_residual_blocks
        )
        
        self.discriminator = Discriminator(
            text_embedding_dim=text_embedding_dim,
            ndf=ndf,
            num_channels=num_channels,
            image_size=image_size
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
            # Repeat text embeddings for multiple samples
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
    """
    Custom weights initialization for better training
    """
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
    # Test the model architecture
    print("Testing Text-to-Image GAN Architecture\n")
    
    # Model parameters
    batch_size = 4
    latent_dim = 100
    text_embedding_dim = 768
    image_size = 64
    num_channels = 3
    
    # Create model
    model = TextToImageGAN(
        latent_dim=latent_dim,
        text_embedding_dim=text_embedding_dim,
        image_size=image_size,
        num_channels=num_channels
    )
    
    # Apply weight initialization
    model.generator.apply(weights_init)
    model.discriminator.apply(weights_init)
    
    # Test forward pass
    noise = torch.randn(batch_size, latent_dim)
    text_embedding = torch.randn(batch_size, text_embedding_dim)
    fake_images = torch.randn(batch_size, num_channels, image_size, image_size)
    
    print("Testing Generator...")
    generated = model.generator(noise, text_embedding)
    print(f"  Input noise shape: {noise.shape}")
    print(f"  Input text embedding shape: {text_embedding.shape}")
    print(f"  Generated images shape: {generated.shape}")
    
    print("\nTesting Discriminator...")
    validity = model.discriminator(fake_images, text_embedding)
    print(f"  Input images shape: {fake_images.shape}")
    print(f"  Output validity shape: {validity.shape}")
    
    # Count parameters
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    total_params = gen_params + disc_params
    
    print(f"\nModel Parameters:")
    print(f"  Generator: {gen_params:,}")
    print(f"  Discriminator: {disc_params:,}")
    print(f"  Total: {total_params:,}")
    
    print("\n✅ Architecture test completed successfully!")
