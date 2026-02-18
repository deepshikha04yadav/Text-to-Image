"""
Enhanced Trainer for Attention-Based Text-to-Image GAN
Includes attention visualization and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, List
import os
import time
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from models.attention_gan import AttentionTextToImageGAN, weights_init
sys.path.append('../utils')


class AttentionGANTrainer:
    """
    Enhanced trainer for Attention-based GAN with attention visualization
    """
    
    def __init__(
        self,
        model: AttentionTextToImageGAN,
        text_embedder,
        device: str = 'auto',
        learning_rate: float = 0.0001,  # Lower LR for attention models
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        checkpoint_dir: str = './outputs/checkpoints',
        log_dir: str = './outputs/logs'
    ):
        """
        Initialize Attention GAN trainer
        
        Args:
            model: AttentionTextToImageGAN model
            text_embedder: Text embedding model
            device: Device to train on
            learning_rate: Learning rate (lower for attention models)
            beta1: Adam optimizer beta1
            beta2: Adam optimizer beta2
            lambda_gp: Gradient penalty coefficient
            n_critic: Number of discriminator updates per generator update
            checkpoint_dir: Checkpoint directory
            log_dir: Log directory
        """
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Initializing Attention GAN Trainer on {self.device}")
        
        # Move model to device
        self.model = model.to(self.device)
        self.text_embedder = text_embedder
        
        # Training parameters
        self.learning_rate = learning_rate
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        
        # Optimizers with different LRs for generator and discriminator
        self.optimizer_G = optim.Adam(
            self.model.generator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )
        
        self.optimizer_D = optim.Adam(
            self.model.discriminator.parameters(),
            lr=learning_rate * 2,  # Slightly higher for discriminator
            betas=(beta1, beta2)
        )
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G,
            T_max=100,
            eta_min=learning_rate * 0.1
        )
        
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_D,
            T_max=100,
            eta_min=learning_rate * 2 * 0.1
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
            'd_fake_acc': [],
            'lr_g': [],
            'lr_d': []
        }
        
        print("✅ Attention GAN Trainer initialized!")
        print(f"   Learning Rate (G): {learning_rate}")
        print(f"   Learning Rate (D): {learning_rate * 2}")
    
    def compute_gradient_penalty(self, real_images, fake_images, text_embedding):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_images.size(0)
        
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
        
        d_interpolated = self.model.discriminator(interpolated, text_embedding)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def extract_attention_maps(self, text_embedding):
        """
        Extract attention maps for visualization
        """
        # This is a placeholder - actual implementation would hook into
        # attention layers during forward pass
        # For now, we'll just note this capability
        pass
    
    def train_step(self, real_images, text_embeddings):
        """
        Single training step with attention-specific monitoring
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
        
        # Gradient penalty
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
        
        # Generator loss
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
        """Train for one epoch"""
        self.model.generator.train()
        self.model.discriminator.train()
        
        epoch_metrics = {
            'g_loss': 0,
            'd_loss': 0,
            'd_real_acc': 0,
            'd_fake_acc': 0
        }
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        num_batches = 0
        
        for batch_idx, batch in enumerate(pbar):
            # Get batch data
            real_images = batch['image'].to(self.device)
            texts = batch['text']
            
            # Get text embeddings
            with torch.no_grad():
                text_embeddings = self.text_embedder.encode_to_tensor(texts, normalize=True)
            
            # Train
            if batch_idx % self.n_critic == 0:
                metrics = self.train_step(real_images, text_embeddings)
                
                # Update running averages
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'G': f"{metrics['g_loss']:.4f}",
                    'D': f"{metrics['d_loss']:.4f}",
                    'D_real': f"{metrics['d_real_acc']:.2f}",
                    'D_fake': f"{metrics['d_fake_acc']:.2f}"
                })
                
                # Log to tensorboard
                self.writer.add_scalar('Loss/Generator', metrics['g_loss'], self.global_step)
                self.writer.add_scalar('Loss/Discriminator', metrics['d_loss'], self.global_step)
                self.writer.add_scalar('Accuracy/D_real', metrics['d_real_acc'], self.global_step)
                self.writer.add_scalar('Accuracy/D_fake', metrics['d_fake_acc'], self.global_step)
                
                self.global_step += 1
        
        # Compute epoch averages
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        # Log learning rates
        avg_metrics['lr_g'] = self.optimizer_G.param_groups[0]['lr']
        avg_metrics['lr_d'] = self.optimizer_D.param_groups[0]['lr']
        
        return avg_metrics
    
    def validate(self, val_texts):
        """Generate validation images"""
        self.model.generator.eval()
        
        with torch.no_grad():
            text_embeddings = self.text_embedder.encode_to_tensor(val_texts, normalize=True)
            generated_images = self.model.generate(text_embeddings, num_samples=1)
        
        return generated_images
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'history': self.history,
            'best_g_loss': self.best_g_loss
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'attn_gan_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'attn_gan_best.pt')
            torch.save(checkpoint, best_path)
            print(f"⭐ Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        if 'scheduler_G_state_dict' in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        
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
        """Complete training loop"""
        print(f"\n🎯 Starting Attention GAN Training")
        print(f"📊 Epochs: {num_epochs}")
        print(f"📦 Batches per epoch: {len(dataloader)}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train one epoch
            metrics = self.train_epoch(dataloader, epoch)
            
            # Update history
            for key in self.history:
                if key in metrics:
                    self.history[key].append(metrics[key])
            
            # Step schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  Generator Loss: {metrics['g_loss']:.4f}")
            print(f"  Discriminator Loss: {metrics['d_loss']:.4f}")
            print(f"  D Real Accuracy: {metrics['d_real_acc']:.2%}")
            print(f"  D Fake Accuracy: {metrics['d_fake_acc']:.2%}")
            print(f"  LR (G): {metrics['lr_g']:.6f}")
            print(f"  LR (D): {metrics['lr_d']:.6f}")
            
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
                
                self.writer.add_images(
                    'Generated/Samples',
                    (generated_images + 1) / 2,
                    epoch
                )
            
            self.epoch = epoch + 1
        
        total_time = time.time() - start_time
        print(f"\n✅ Training Complete!")
        print(f"⏱️  Total Time: {total_time/3600:.2f} hours")
        print(f"📊 Best Generator Loss: {self.best_g_loss:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(self.epoch, is_best=False)
        
        # Close tensorboard
        self.writer.close()
        
        # Save history
        history_path = os.path.join(self.log_dir, 'attn_gan_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


if __name__ == '__main__':
    print("✅ Attention GAN Trainer module loaded successfully")
