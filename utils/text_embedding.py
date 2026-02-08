"""
Text Embedding Module
Handles conversion of text to dense vector representations using pre-trained models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Dict
from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer
import warnings
warnings.filterwarnings('ignore')


class TextEmbedder:
    """
    Text embedding using pre-trained transformer models (BERT, CLIP, etc.)
    """
    
    def __init__(self, 
                 model_name: str = 'openai/clip-vit-base-patch32',
                 device: str = None,
                 max_length: int = 77):
        """
        Initialize text embedder
        
        Args:
            model_name: Name of pre-trained model
                      - 'openai/clip-vit-base-patch32' (recommended for images)
                      - 'sentence-transformers/all-MiniLM-L6-v2' (general purpose)
                      - 'bert-base-uncased' (BERT)
            device: Device to run model on ('cuda' or 'cpu')
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Loading text embedding model: {model_name}")
        print(f"📍 Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            if 'clip' in model_name.lower():
                self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
                self.model = CLIPTextModel.from_pretrained(model_name)
                self.is_clip = True
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.is_clip = False
            
            self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size
            print(f"✅ Model loaded successfully!")
            print(f"📊 Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  Initializing with random embeddings instead")
            self.tokenizer = None
            self.model = None
            self.embedding_dim = 768
            self.is_clip = False
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings
        
        Args:
            model_output: Model output
            attention_mask: Attention mask
            
        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output[0]  # First element is hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            normalize: Normalize embeddings to unit length
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model is None:
            # Return random embeddings if model not loaded
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            if normalize:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Get embeddings
                if self.is_clip:
                    # For CLIP, use the pooled output
                    outputs = self.model(**encoded_input)
                    embeddings = outputs.pooler_output
                else:
                    # For BERT-like models, use mean pooling
                    outputs = self.model(**encoded_input)
                    embeddings = self.mean_pooling(outputs, encoded_input['attention_mask'])
                
                # Normalize if requested
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single text to embedding
        
        Args:
            text: Input text
            normalize: Normalize embedding
            
        Returns:
            Embedding vector
        """
        embeddings = self.encode([text], normalize=normalize)
        return embeddings[0]
    
    def encode_to_tensor(self, texts: Union[str, List[str]], normalize: bool = True) -> torch.Tensor:
        """
        Encode texts directly to PyTorch tensor
        
        Args:
            texts: Single text or list of texts
            normalize: Normalize embeddings
            
        Returns:
            Tensor of embeddings
        """
        embeddings = self.encode(texts, normalize=normalize)
        return torch.from_numpy(embeddings).to(self.device)
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dim


class TextEmbeddingCache:
    """
    Cache for storing computed text embeddings
    Useful for avoiding recomputation during training
    """
    
    def __init__(self, cache_size: int = 10000):
        """
        Initialize embedding cache
        
        Args:
            cache_size: Maximum number of embeddings to cache
        """
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = {}
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache
        
        Args:
            text: Input text
            
        Returns:
            Cached embedding or None
        """
        if text in self.cache:
            self.access_count[text] = self.access_count.get(text, 0) + 1
            return self.cache[text]
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """
        Store embedding in cache
        
        Args:
            text: Input text
            embedding: Computed embedding
        """
        if len(self.cache) >= self.cache_size:
            # Remove least frequently accessed item
            if self.access_count:
                least_accessed = min(self.access_count, key=self.access_count.get)
                del self.cache[least_accessed]
                del self.access_count[least_accessed]
        
        self.cache[text] = embedding
        self.access_count[text] = 1
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_count.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        if not self.access_count:
            return {"size": 0, "avg_access": 0, "max_access": 0}
        
        accesses = list(self.access_count.values())
        return {
            "size": len(self.cache),
            "avg_access": sum(accesses) / len(accesses),
            "max_access": max(accesses),
            "total_accesses": sum(accesses)
        }


class CachedTextEmbedder(TextEmbedder):
    """
    Text embedder with built-in caching
    """
    
    def __init__(self, *args, cache_size: int = 10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = TextEmbeddingCache(cache_size=cache_size)
        print(f"✅ Embedding cache enabled (size: {cache_size})")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Encode texts with caching
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                embeddings.append((i, cached_embedding))
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Encode uncached texts
        if texts_to_encode:
            new_embeddings = super().encode(texts_to_encode, batch_size=batch_size, normalize=normalize)
            
            # Cache new embeddings
            for text, embedding in zip(texts_to_encode, new_embeddings):
                self.cache.set(text, embedding)
            
            # Add to results
            for idx, embedding in zip(indices_to_encode, new_embeddings):
                embeddings.append((idx, embedding))
        
        # Sort by original order and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache.stats()


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("Text Embedding Module Test")
    print("=" * 60)
    
    # Test CLIP embedder
    print("\n1. Testing CLIP Text Embedder\n")
    
    try:
        embedder = TextEmbedder(model_name='openai/clip-vit-base-patch32')
        
        sample_texts = [
            "A beautiful sunset over the ocean",
            "A cute dog playing in the park",
            "Modern architecture building with glass windows"
        ]
        
        print(f"Encoding {len(sample_texts)} texts...")
        embeddings = embedder.encode(sample_texts)
        
        print(f"✓ Embeddings shape: {embeddings.shape}")
        print(f"✓ First embedding (first 10 dims): {embeddings[0][:10]}")
        print(f"✓ Embedding norm: {np.linalg.norm(embeddings[0]):.4f}")
        
    except Exception as e:
        print(f"⚠️ CLIP test failed: {e}")
    
    # Test cached embedder
    print("\n\n2. Testing Cached Text Embedder\n")
    
    try:
        cached_embedder = CachedTextEmbedder(cache_size=100)
        
        # First encoding
        text = "A beautiful landscape"
        print(f"Encoding: '{text}'")
        emb1 = cached_embedder.encode_single(text)
        print(f"✓ First encoding: {emb1.shape}")
        
        # Second encoding (should use cache)
        emb2 = cached_embedder.encode_single(text)
        print(f"✓ Second encoding (cached): {emb2.shape}")
        print(f"✓ Embeddings match: {np.allclose(emb1, emb2)}")
        
        # Cache stats
        stats = cached_embedder.get_cache_stats()
        print(f"\n📊 Cache Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"⚠️ Cached embedder test failed: {e}")
    
    print("\n" + "=" * 60)
