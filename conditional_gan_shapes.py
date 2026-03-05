"""
CONDITIONAL GAN (CGAN) - Shape Generator
Generates basic shapes (circle, square, triangle, star, etc.) from text labels

This demonstrates the core concept of conditional GANs:
- Generator receives both noise AND label
- Discriminator receives both image AND label
- Label guides what shape to generate
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
from tqdm import tqdm
import random


# ============================================================================
# SHAPE DATASET GENERATOR
# ============================================================================

class ShapeGenerator:
    """
    Generate synthetic shapes programmatically
    """
    
    SHAPES = ['circle', 'square', 'triangle', 'star', 'diamond', 'hexagon']
    
    @staticmethod
    def create_circle(size=64, color=255):
        """Create a circle image"""
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        margin = size // 6
        draw.ellipse([margin, margin, size-margin, size-margin], fill=color)
        
        return np.array(img)
    
    @staticmethod
    def create_square(size=64, color=255):
        """Create a square image"""
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        margin = size // 6
        draw.rectangle([margin, margin, size-margin, size-margin], fill=color)
        
        return np.array(img)
    
    @staticmethod
    def create_triangle(size=64, color=255):
        """Create a triangle image"""
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        margin = size // 6
        points = [
            (size // 2, margin),  # Top
            (margin, size - margin),  # Bottom left
            (size - margin, size - margin)  # Bottom right
        ]
        draw.polygon(points, fill=color)
        
        return np.array(img)
    
    @staticmethod
    def create_star(size=64, color=255):
        """Create a 5-point star"""
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        # Calculate star points
        center_x, center_y = size // 2, size // 2
        outer_radius = size // 2 - size // 8
        inner_radius = outer_radius // 2
        
        points = []
        for i in range(10):
            angle = np.pi / 2 + i * np.pi / 5
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = center_x + radius * np.cos(angle)
            y = center_y - radius * np.sin(angle)
            points.append((x, y))
        
        draw.polygon(points, fill=color)
        
        return np.array(img)
    
    @staticmethod
    def create_diamond(size=64, color=255):
        """Create a diamond (rotated square)"""
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        margin = size // 6
        points = [
            (size // 2, margin),  # Top
            (size - margin, size // 2),  # Right
            (size // 2, size - margin),  # Bottom
            (margin, size // 2)  # Left
        ]
        draw.polygon(points, fill=color)
        
        return np.array(img)
    
    @staticmethod
    def create_hexagon(size=64, color=255):
        """Create a hexagon"""
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size // 2, size // 2
        radius = size // 2 - size // 8
        
        points = []
        for i in range(6):
            angle = np.pi / 3 * i
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append((x, y))
        
        draw.polygon(points, fill=color)
        
        return np.array(img)
    
    @classmethod
    def generate_shape(cls, shape_name, size=64, add_noise=True):
        """
        Generate a shape by name
        
        Args:
            shape_name: Name of shape
            size: Image size
            add_noise: Add random variations
            
        Returns:
            numpy array of shape image
        """
        # Random color variation
        color = 255 if not add_noise else random.randint(200, 255)
        
        # Create shape
        if shape_name == 'circle':
            img = cls.create_circle(size, color)
        elif shape_name == 'square':
            img = cls.create_square(size, color)
        elif shape_name == 'triangle':
            img = cls.create_triangle(size, color)
        elif shape_name == 'star':
            img = cls.create_star(size, color)
        elif shape_name == 'diamond':
            img = cls.create_diamond(size, color)
        elif shape_name == 'hexagon':
            img = cls.create_hexagon(size, color)
        else:
            raise ValueError(f"Unknown shape: {shape_name}")
        
        # Add random noise if requested
        if add_noise:
            noise = np.random.normal(0, 5, img.shape)
            img = np.clip(img + noise, 0, 255)
        
        return img.astype(np.float32) / 255.0  # Normalize to [0, 1]


class ShapeDataset(Dataset):
    """
    Dataset of synthetic shapes
    """
    
    def __init__(self, num_samples=10000, size=64):
        """
        Initialize dataset
        
        Args:
            num_samples: Number of samples
            size: Image size
        """
        self.num_samples = num_samples
        self.size = size
        self.shapes = ShapeGenerator.SHAPES
        self.shape_to_idx = {shape: i for i, shape in enumerate(self.shapes)}
        
        print(f"📊 ShapeDataset initialized:")
        print(f"   Samples: {num_samples}")
        print(f"   Image size: {size}x{size}")
        print(f"   Shapes ({len(self.shapes)}): {', '.join(self.shapes)}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random shape
        shape_name = random.choice(self.shapes)
        shape_label = self.shape_to_idx[shape_name]
        
        # Generate shape
        image = ShapeGenerator.generate_shape(shape_name, self.size, add_noise=True)
        
        # Convert to tensor
        image = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
        
        return {
            'image': image,
            'label': shape_label,
            'shape_name': shape_name
        }


# ============================================================================
# CONDITIONAL GAN MODELS
# ============================================================================

class ConditionalGenerator(nn.Module):
    """
    Conditional Generator: (noise + label) → image
    
    Key concept: Label is embedded and concatenated with noise
    """
    
    def __init__(self, 
                 latent_dim=100,
                 num_classes=6,
                 image_size=64,
                 num_channels=1):
        """
        Initialize generator
        
        Args:
            latent_dim: Noise vector dimension
            num_classes: Number of shape classes
            image_size: Output image size
            num_channels: Number of image channels (1 for grayscale)
        """
        super(ConditionalGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 50)
        
        # Input: noise (100) + label embedding (50) = 150
        input_dim = latent_dim + 50
        
        # Generator network
        self.model = nn.Sequential(
            # 150 → 256
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 → 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 → 1024
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 1024 → image_size^2
            nn.Linear(1024, num_channels * image_size * image_size),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        self.num_channels = num_channels
    
    def forward(self, noise, labels):
        """
        Forward pass
        
        Args:
            noise: [batch_size, latent_dim]
            labels: [batch_size] - shape class indices
            
        Returns:
            Generated images [batch_size, channels, H, W]
        """
        # Embed labels
        label_emb = self.label_embedding(labels)  # [batch_size, 50]
        
        # Concatenate noise and label embedding
        gen_input = torch.cat([noise, label_emb], dim=1)  # [batch_size, 150]
        
        # Generate
        img = self.model(gen_input)
        
        # Reshape to image
        img = img.view(img.size(0), self.num_channels, self.image_size, self.image_size)
        
        return img


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator: (image + label) → real/fake
    
    Key concept: Label is embedded and concatenated with image features
    """
    
    def __init__(self, 
                 num_classes=6,
                 image_size=64,
                 num_channels=1):
        """
        Initialize discriminator
        
        Args:
            num_classes: Number of shape classes
            image_size: Input image size
            num_channels: Number of image channels
        """
        super(ConditionalDiscriminator, self).__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 50)
        
        # Input: flattened image + label embedding
        input_dim = num_channels * image_size * image_size + 50
        
        # Discriminator network
        self.model = nn.Sequential(
            # input_dim → 512
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # 512 → 256
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # 256 → 1 (real/fake)
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, images, labels):
        """
        Forward pass
        
        Args:
            images: [batch_size, channels, H, W]
            labels: [batch_size] - shape class indices
            
        Returns:
            Predictions [batch_size, 1] - probability of being real
        """
        # Flatten images
        img_flat = images.view(images.size(0), -1)
        
        # Embed labels
        label_emb = self.label_embedding(labels)
        
        # Concatenate image and label
        disc_input = torch.cat([img_flat, label_emb], dim=1)
        
        # Classify
        validity = self.model(disc_input)
        
        return validity


# ============================================================================
# TRAINER
# ============================================================================

class CGANTrainer:
    """
    Trainer for Conditional GAN
    """
    
    def __init__(self,
                 generator,
                 discriminator,
                 device='cuda',
                 lr_g=0.0002,
                 lr_d=0.0002):
        """
        Initialize trainer
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            device: Device to train on
            lr_g: Generator learning rate
            lr_d: Discriminator learning rate
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        # Optimizers
        self.optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        # Loss
        self.adversarial_loss = nn.BCELoss()
        
        print(f"🎓 CGAN Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Generator LR: {lr_g}")
        print(f"   Discriminator LR: {lr_d}")
    
    def train_step(self, real_images, labels):
        """
        Single training step
        
        Args:
            real_images: Real images [batch_size, 1, H, W]
            labels: Shape labels [batch_size]
            
        Returns:
            Dictionary with losses
        """
        batch_size = real_images.size(0)
        
        # Move to device
        real_images = real_images.to(self.device)
        labels = labels.to(self.device)
        
        # Labels for real/fake
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # =================== Train Discriminator ===================
        
        self.optimizer_D.zero_grad()
        
        # Real images
        real_validity = self.discriminator(real_images, labels)
        d_real_loss = self.adversarial_loss(real_validity, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
        fake_images = self.generator(noise, labels)
        fake_validity = self.discriminator(fake_images.detach(), labels)
        d_fake_loss = self.adversarial_loss(fake_validity, fake_labels)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        d_loss.backward()
        self.optimizer_D.step()
        
        # =================== Train Generator ===================
        
        self.optimizer_G.zero_grad()
        
        # Generate images
        noise = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
        fake_images = self.generator(noise, labels)
        
        # Try to fool discriminator
        validity = self.discriminator(fake_images, labels)
        g_loss = self.adversarial_loss(validity, real_labels)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_real': real_validity.mean().item(),
            'd_fake': fake_validity.mean().item()
        }
    
    def train(self, dataloader, num_epochs, save_dir='./outputs/cgan'):
        """
        Train the CGAN
        
        Args:
            dataloader: Training dataloader
            num_epochs: Number of epochs
            save_dir: Directory to save outputs
        """
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{save_dir}/samples', exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"TRAINING CONDITIONAL GAN")
        print(f"{'='*80}")
        print(f"Epochs: {num_epochs}")
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Save dir: {save_dir}")
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                real_images = batch['image']
                labels = batch['label']
                
                # Train step
                losses = self.train_step(real_images, labels)
                
                epoch_g_loss += losses['g_loss']
                epoch_d_loss += losses['d_loss']
                
                # Update progress bar
                pbar.set_postfix({
                    'G': f"{losses['g_loss']:.4f}",
                    'D': f"{losses['d_loss']:.4f}",
                    'D(real)': f"{losses['d_real']:.2f}",
                    'D(fake)': f"{losses['d_fake']:.2f}"
                })
            
            # Epoch summary
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"   G Loss: {avg_g_loss:.4f}")
            print(f"   D Loss: {avg_d_loss:.4f}")
            
            # Save samples every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_samples(epoch + 1, save_dir)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, save_dir)
        
        print(f"\n✅ Training complete!")
    
    def save_samples(self, epoch, save_dir):
        """Save generated samples for all shapes"""
        self.generator.eval()
        
        num_samples = 3
        shapes = ShapeGenerator.SHAPES
        
        fig, axes = plt.subplots(len(shapes), num_samples, figsize=(num_samples*2, len(shapes)*2))
        
        with torch.no_grad():
            for i, shape in enumerate(shapes):
                shape_idx = i
                
                for j in range(num_samples):
                    # Generate
                    noise = torch.randn(1, self.generator.latent_dim, device=self.device)
                    label = torch.tensor([shape_idx], device=self.device)
                    
                    generated = self.generator(noise, label)
                    
                    # Convert to image
                    img = generated[0, 0].cpu().numpy()
                    img = (img + 1) / 2  # [-1, 1] → [0, 1]
                    
                    # Plot
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].axis('off')
                    
                    if j == 0:
                        axes[i, j].set_title(shape, fontsize=12, loc='left')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/samples/epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved samples: epoch_{epoch}.png")
        
        self.generator.train()
    
    def save_checkpoint(self, epoch, save_dir):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict()
        }
        
        path = f'{save_dir}/checkpoints/cgan_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"   💾 Saved checkpoint: cgan_epoch_{epoch}.pt")


# ============================================================================
# GENERATION INTERFACE
# ============================================================================

class ShapeGeneratorInterface:
    """
    Easy-to-use interface for generating shapes
    """
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize interface
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.generator = ConditionalGenerator(
            latent_dim=100,
            num_classes=6,
            image_size=64
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.generator.eval()
        
        self.shapes = ShapeGenerator.SHAPES
        self.shape_to_idx = {shape: i for i, shape in enumerate(self.shapes)}
        
        print(f"✅ ShapeGenerator loaded from {checkpoint_path}")
        print(f"   Available shapes: {', '.join(self.shapes)}")
    
    def generate(self, shape_name, num_samples=1, seed=None):
        """
        Generate shape images
        
        Args:
            shape_name: Name of shape ('circle', 'square', etc.)
            num_samples: Number of samples to generate
            seed: Random seed (optional)
            
        Returns:
            List of numpy arrays
        """
        if shape_name not in self.shapes:
            raise ValueError(f"Unknown shape: {shape_name}. Available: {self.shapes}")
        
        if seed is not None:
            torch.manual_seed(seed)
        
        shape_idx = self.shape_to_idx[shape_name]
        
        images = []
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.generator.latent_dim, device=self.device)
            labels = torch.tensor([shape_idx] * num_samples, device=self.device)
            
            generated = self.generator(noise, labels)
            
            for i in range(num_samples):
                img = generated[i, 0].cpu().numpy()
                img = (img + 1) / 2  # [-1, 1] → [0, 1]
                img = (img * 255).astype(np.uint8)
                images.append(img)
        
        return images
    
    def visualize(self, shape_name, num_samples=4):
        """
        Generate and visualize shapes
        
        Args:
            shape_name: Name of shape
            num_samples: Number of samples
        """
        images = self.generate(shape_name, num_samples)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*2, 2))
        
        for i, (img, ax) in enumerate(zip(images, axes)):
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{shape_name} {i+1}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("CONDITIONAL GAN - SHAPE GENERATOR")
    print("=" * 80)
    
    # Configuration
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    IMAGE_SIZE = 64
    LATENT_DIM = 100
    NUM_CLASSES = 6
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n⚙️  Configuration:")
    print(f"   Device: {device}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   Number of shapes: {NUM_CLASSES}")
    
    # Create dataset
    print(f"\n📊 Creating dataset...")
    dataset = ShapeDataset(num_samples=10000, size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Create models
    print(f"\n🏗️  Creating models...")
    generator = ConditionalGenerator(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE
    )
    
    discriminator = ConditionalDiscriminator(
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE
    )
    
    print(f"   Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"   Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create trainer
    print(f"\n🎓 Creating trainer...")
    trainer = CGANTrainer(generator, discriminator, device=device)
    
    # Train
    print(f"\n🚀 Starting training...")
    trainer.train(dataloader, num_epochs=NUM_EPOCHS)
    
    print(f"\n✅ Complete!")
    print(f"   Checkpoints: ./outputs/cgan/checkpoints/")
    print(f"   Samples: ./outputs/cgan/samples/")