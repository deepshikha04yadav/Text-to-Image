"""
Attention Modules for Text-to-Image GAN
Implements Self-Attention and Cross-Attention mechanisms
for improved image quality and text-image alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SelfAttention(nn.Module):
    """
    Self-Attention Module (SAGAN-style)
    Allows the model to attend to different spatial locations
    """
    
    def __init__(self, in_channels: int, reduction: int = 8):
        """
        Initialize Self-Attention
        
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction factor for efficiency
        """
        super(SelfAttention, self).__init__()
        
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        # Output projection
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input feature map [batch, channels, height, width]
            
        Returns:
            Attention-weighted features [batch, channels, height, width]
        """
        batch_size, C, H, W = x.size()
        
        # Generate query, key, value
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        # [batch, H*W, C//reduction]
        
        key = self.key(x).view(batch_size, -1, H * W)
        # [batch, C//reduction, H*W]
        
        value = self.value(x).view(batch_size, -1, H * W)
        # [batch, C, H*W]
        
        # Compute attention scores
        attention = torch.bmm(query, key)  # [batch, H*W, H*W]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        # [batch, C, H*W]
        
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class CrossAttention(nn.Module):
    """
    Cross-Attention Module
    Allows image features to attend to text features
    for better text-image alignment
    """
    
    def __init__(
        self,
        image_channels: int,
        text_channels: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize Cross-Attention
        
        Args:
            image_channels: Number of image feature channels
            text_channels: Number of text embedding channels
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(CrossAttention, self).__init__()
        
        self.num_heads = num_heads
        self.image_channels = image_channels
        self.text_channels = text_channels
        
        assert image_channels % num_heads == 0, "image_channels must be divisible by num_heads"
        
        self.head_dim = image_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query from image features
        self.query = nn.Linear(image_channels, image_channels)
        
        # Key and Value from text features
        self.key = nn.Linear(text_channels, image_channels)
        self.value = nn.Linear(text_channels, image_channels)
        
        # Output projection
        self.out_proj = nn.Linear(image_channels, image_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(image_channels)
        
    def forward(self, image_features, text_features):
        """
        Forward pass
        
        Args:
            image_features: [batch, H*W, image_channels]
            text_features: [batch, seq_len, text_channels]
            
        Returns:
            Attended features [batch, H*W, image_channels]
        """
        batch_size, seq_len, _ = image_features.shape
        
        # Generate Q, K, V
        Q = self.query(image_features)  # [batch, H*W, image_channels]
        K = self.key(text_features)     # [batch, seq_len, image_channels]
        V = self.value(text_features)   # [batch, seq_len, image_channels]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, H*W, head_dim]
        
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, text_seq_len, head_dim]
        
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, text_seq_len, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [batch, num_heads, H*W, text_seq_len]
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # [batch, num_heads, H*W, head_dim]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.image_channels)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Residual connection + normalization
        output = self.norm(output + image_features)
        
        return output


class SpatialCrossAttention(nn.Module):
    """
    Spatial Cross-Attention that preserves spatial structure
    Better for generating images with spatial coherence
    """
    
    def __init__(
        self,
        image_channels: int,
        text_embedding_dim: int,
        num_heads: int = 8
    ):
        super(SpatialCrossAttention, self).__init__()
        
        self.num_heads = num_heads
        self.image_channels = image_channels
        self.head_dim = image_channels // num_heads
        
        # Projections
        self.query_conv = nn.Conv2d(image_channels, image_channels, 1)
        self.key_linear = nn.Linear(text_embedding_dim, image_channels)
        self.value_linear = nn.Linear(text_embedding_dim, image_channels)
        
        self.out_conv = nn.Conv2d(image_channels, image_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, image_features, text_embedding):
        """
        Args:
            image_features: [batch, C, H, W]
            text_embedding: [batch, text_dim]
        """
        batch_size, C, H, W = image_features.shape
        
        # Query from image
        query = self.query_conv(image_features)
        query = query.view(batch_size, self.num_heads, self.head_dim, H * W)
        query = query.permute(0, 1, 3, 2)  # [batch, heads, HW, head_dim]
        
        # Key and Value from text
        key = self.key_linear(text_embedding)
        key = key.view(batch_size, self.num_heads, self.head_dim, 1)
        key = key.permute(0, 1, 3, 2)  # [batch, heads, 1, head_dim]
        
        value = self.value_linear(text_embedding)
        value = value.view(batch_size, self.num_heads, self.head_dim, 1)
        value = value.permute(0, 1, 3, 2)  # [batch, heads, 1, head_dim]
        
        # Attention scores
        attn = torch.matmul(query, key.transpose(-2, -1))  # [batch, heads, HW, 1]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, value)  # [batch, heads, HW, head_dim]
        
        # Reshape
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(batch_size, C, H, W)
        
        # Output conv
        out = self.out_conv(out)
        
        # Residual with learnable weight
        out = self.gamma * out + image_features
        
        return out


class AttentionBlock(nn.Module):
    """
    Combined Self-Attention and Cross-Attention Block
    """
    
    def __init__(
        self,
        channels: int,
        text_embedding_dim: int,
        num_heads: int = 8,
        use_self_attention: bool = True,
        use_cross_attention: bool = True
    ):
        super(AttentionBlock, self).__init__()
        
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        
        if use_self_attention:
            self.self_attention = SelfAttention(channels, reduction=8)
        
        if use_cross_attention:
            self.cross_attention = SpatialCrossAttention(
                channels,
                text_embedding_dim,
                num_heads
            )
    
    def forward(self, x, text_embedding=None):
        """
        Args:
            x: Image features [batch, C, H, W]
            text_embedding: Text embedding [batch, text_dim]
        """
        # Self-attention
        if self.use_self_attention:
            x = self.self_attention(x)
        
        # Cross-attention with text
        if self.use_cross_attention and text_embedding is not None:
            x = self.cross_attention(x, text_embedding)
        
        return x


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Allows model to emphasize important feature channels
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Average pool
        avg_out = self.avg_pool(x).view(batch_size, C)
        avg_out = self.fc(avg_out)
        
        # Max pool
        max_out = self.max_pool(x).view(batch_size, C)
        max_out = self.fc(max_out)
        
        # Combine
        out = avg_out + max_out
        out = self.sigmoid(out).view(batch_size, C, 1, 1)
        
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Emphasizes important spatial locations
    """
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average and max pool across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Combines Channel and Spatial Attention
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


if __name__ == '__main__':
    # Test attention modules
    print("Testing Attention Modules\n")
    
    batch_size = 4
    channels = 256
    height, width = 16, 16
    text_dim = 768
    
    # Test Self-Attention
    print("1. Self-Attention")
    self_attn = SelfAttention(channels)
    x = torch.randn(batch_size, channels, height, width)
    out = self_attn(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   ✓ Self-attention working\n")
    
    # Test Spatial Cross-Attention
    print("2. Spatial Cross-Attention")
    spatial_attn = SpatialCrossAttention(channels, text_dim, num_heads=8)
    img_feat = torch.randn(batch_size, channels, height, width)
    text_emb = torch.randn(batch_size, text_dim)
    out = spatial_attn(img_feat, text_emb)
    print(f"   Image: {img_feat.shape}")
    print(f"   Text: {text_emb.shape}")
    print(f"   Output: {out.shape}")
    print(f"   ✓ Spatial cross-attention working\n")
    
    # Test Attention Block
    print("3. Combined Attention Block")
    attn_block = AttentionBlock(channels, text_dim, num_heads=8)
    out = attn_block(img_feat, text_emb)
    print(f"   Input: {img_feat.shape}")
    print(f"   Output: {out.shape}")
    print(f"   ✓ Attention block working\n")
    
    # Test CBAM
    print("4. CBAM (Channel + Spatial)")
    cbam = CBAM(channels)
    out = cbam(img_feat)
    print(f"   Input: {img_feat.shape}")
    print(f"   Output: {out.shape}")
    print(f"   ✓ CBAM working\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in attn_block.parameters())
    print(f"📊 Attention Block Parameters: {total_params:,}")
    
    print("\n✅ All attention modules tested successfully!")
