"""
Training Pipeline for Text-to-Image GAN
Includes training loop, checkpointing, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
import os
import time
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np

import sys
sys.path.append('..')
from models.text_to_image_gan import TextToImageGAN, weights_init
from utils.text_embedding import TextEmbedder


class GANTrainer:
    """
    Comprehensive trainer for Text-to-Image GAN
    """
    
    def __init__(
        self,
        model: TextToImageGAN,
        text_embedder: TextEmbedder,
        device: str = 'auto',
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        checkpoint_dir: str = './outputs/checkpoints',
        log_dir: str = './outputs/logs'
    ):
        """
        Initialize GAN trainer
        
        Args:
            model: TextToImageGAN model
            text_embedder: Text embedding model
            device: Device to train on
            learning_rate: Learning rate for optimizers
            beta1: Adam optimizer beta1 parameter
            beta2: Adam optimizer beta2 parameter
            lambda_gp: Gradient penalty coefficient (for WGAN-GP)
            n_critic: Number of discriminator updates per generator update
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
        """
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Initializing GAN Trainer on {self.device}")
        
        # Move model to device
        self.model = model.to(self.device)
        self.text_embedder = text_embedder
        
        # Training parameters
        self.learning_rate = learning_rate
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.model.generator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )
        
        self.optimizer_D = optim.Adam(
            self.model.discriminator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Setup directories
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_g_loss = float('inf')
        
        # History
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': []
        }
        
        print("✅ Trainer initialized successfully!")
    
    def compute_gradient_penalty(self, real_images, fake_images, text_embedding):
        """
        Compute gradient penalty for WGAN-GP
        
        Args:
            real_images: Real images
            fake_images: Generated images
            text_embedding: Text embeddings
            
        Returns:
            Gradient penalty value
        """
        batch_size = real_images.size(0)
        
        # Random weight term for interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        # Interpolate between real and fake images
        interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
        
        # Get discriminator output for interpolated images
        d_interpolated = self.model.discriminator(interpolated, text_embedding)
        
        # Get gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Compute penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_images, text_embeddings):
        """
        Single training step
        
        Args:
            real_images: Batch of real images
            text_embeddings: Corresponding text embeddings
            
        Returns:
            Dictionary of losses and metrics
        """
        batch_size = real_images.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # ================== Train Discriminator ==================
        self.optimizer_D.zero_grad()
        
        # Real images
        real_validity = self.model.discriminator(real_images, text_embeddings)
        d_real_loss = self.criterion(real_validity, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, self.model.latent_dim, device=self.device)
        fake_images = self.model.generator(noise, text_embeddings)
        fake_validity = self.model.discriminator(fake_images.detach(), text_embeddings)
        d_fake_loss = self.criterion(fake_validity, fake_labels)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        # Gradient penalty (optional, for WGAN-GP)
        if self.lambda_gp > 0:
            gp = self.compute_gradient_penalty(real_images, fake_images.detach(), text_embeddings)
            d_loss = d_loss + self.lambda_gp * gp
        
        d_loss.backward()
        self.optimizer_D.step()
        
        # ================== Train Generator ==================
        self.optimizer_G.zero_grad()
        
        # Generate images
        noise = torch.randn(batch_size, self.model.latent_dim, device=self.device)
        fake_images = self.model.generator(noise, text_embeddings)
        
        # Generator loss (fool discriminator)
        fake_validity = self.model.discriminator(fake_images, text_embeddings)
        g_loss = self.criterion(fake_validity, real_labels)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        # Compute accuracies
        d_real_acc = (real_validity > 0.5).float().mean().item()
        d_fake_acc = (fake_validity < 0.5).float().mean().item()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_real_acc': d_real_acc,
            'd_fake_acc': d_fake_acc
        }
    
    def train_epoch(self, dataloader, epoch):
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average losses for the epoch
        """
        self.model.generator.train()
        self.model.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_d_real_acc = 0
        epoch_d_fake_acc = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Get batch data
            real_images = batch['image'].to(self.device)
            texts = batch['text']
            
            # Get text embeddings
            with torch.no_grad():
                text_embeddings = self.text_embedder.encode_to_tensor(texts, normalize=True)
            
            # Train discriminator n_critic times
            if batch_idx % self.n_critic == 0:
                metrics = self.train_step(real_images, text_embeddings)
                
                # Update running averages
                epoch_g_loss += metrics['g_loss']
                epoch_d_loss += metrics['d_loss']
                epoch_d_real_acc += metrics['d_real_acc']
                epoch_d_fake_acc += metrics['d_fake_acc']
                
                # Update progress bar
                pbar.set_postfix({
                    'G_loss': f"{metrics['g_loss']:.4f}",
                    'D_loss': f"{metrics['d_loss']:.4f}",
                    'D_real_acc': f"{metrics['d_real_acc']:.2f}",
                    'D_fake_acc': f"{metrics['d_fake_acc']:.2f}"
                })
                
                # Log to tensorboard
                self.writer.add_scalar('Loss/Generator', metrics['g_loss'], self.global_step)
                self.writer.add_scalar('Loss/Discriminator', metrics['d_loss'], self.global_step)
                self.writer.add_scalar('Accuracy/D_real', metrics['d_real_acc'], self.global_step)
                self.writer.add_scalar('Accuracy/D_fake', metrics['d_fake_acc'], self.global_step)
                
                self.global_step += 1
        
        # Compute epoch averages
        num_batches = len(dataloader) // self.n_critic
        avg_metrics = {
            'g_loss': epoch_g_loss / num_batches,
            'd_loss': epoch_d_loss / num_batches,
            'd_real_acc': epoch_d_real_acc / num_batches,
            'd_fake_acc': epoch_d_fake_acc / num_batches
        }
        
        return avg_metrics
    
    def validate(self, val_texts):
        """
        Generate validation images
        
        Args:
            val_texts: List of validation texts
            
        Returns:
            Generated images
        """
        self.model.generator.eval()
        
        with torch.no_grad():
            text_embeddings = self.text_embedder.encode_to_tensor(val_texts, normalize=True)
            generated_images = self.model.generate(text_embeddings, num_samples=1)
        
        return generated_images
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'history': self.history,
            'best_g_loss': self.best_g_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"⭐ Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint['history']
        self.best_g_loss = checkpoint['best_g_loss']
        
        print(f"✅ Checkpoint loaded from epoch {self.epoch}")
    
    def train(
        self,
        dataloader,
        num_epochs: int,
        val_texts: Optional[list] = None,
        save_interval: int = 5,
        sample_interval: int = 1
    ):
        """
        Complete training loop
        
        Args:
            dataloader: Training data loader
            num_epochs: Number of epochs to train
            val_texts: Validation texts for image generation
            save_interval: Save checkpoint every N epochs
            sample_interval: Generate sample images every N epochs
        """
        print(f"\n🎯 Starting Training")
        print(f"📊 Epochs: {num_epochs}")
        print(f"📦 Batches per epoch: {len(dataloader)}")
        print(f"💾 Checkpoint interval: {save_interval} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train one epoch
            metrics = self.train_epoch(dataloader, epoch)
            
            # Update history
            self.history['g_loss'].append(metrics['g_loss'])
            self.history['d_loss'].append(metrics['d_loss'])
            self.history['d_real_acc'].append(metrics['d_real_acc'])
            self.history['d_fake_acc'].append(metrics['d_fake_acc'])
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  Generator Loss: {metrics['g_loss']:.4f}")
            print(f"  Discriminator Loss: {metrics['d_loss']:.4f}")
            print(f"  D Real Accuracy: {metrics['d_real_acc']:.2%}")
            print(f"  D Fake Accuracy: {metrics['d_fake_acc']:.2%}")
            
            # Save checkpoint
            is_best = metrics['g_loss'] < self.best_g_loss
            if is_best:
                self.best_g_loss = metrics['g_loss']
            
            if (epoch + 1) % save_interval == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best=is_best)
            
            # Generate validation samples
            if val_texts and (epoch + 1) % sample_interval == 0:
                print(f"\n🎨 Generating validation samples...")
                generated_images = self.validate(val_texts)
                
                # Save to tensorboard
                self.writer.add_images(
                    'Generated/Samples',
                    (generated_images + 1) / 2,  # Denormalize from [-1,1] to [0,1]
                    epoch
                )
            
            self.epoch = epoch + 1
        
        total_time = time.time() - start_time
        print(f"\n✅ Training Complete!")
        print(f"⏱️  Total Time: {total_time/3600:.2f} hours")
        print(f"📊 Best Generator Loss: {self.best_g_loss:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(self.epoch, is_best=False)
        
        # Close tensorboard writer
        self.writer.close()
        
        # Save training history
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"📁 Training history saved: {history_path}")


if __name__ == '__main__':
    print("GAN Training Pipeline Test\n")
    
    # This would be used with actual data
    print("✓ Training pipeline module loaded successfully")
    print("✓ Ready for training with real dataset")
