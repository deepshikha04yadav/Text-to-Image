"""
ADVANCED TEXT PREPROCESSING & ENCODING SYSTEM
Using Hugging Face Transformers for Text-to-Image Models

Features:
- Multiple encoder support (CLIP, BERT, T5, GPT-2)
- Advanced preprocessing (cleaning, normalization, augmentation)
- Tokenization with multiple strategies
- Embedding generation and caching
- Batch processing
- Quality analysis
"""

import torch
import torch.nn as nn
from transformers import (
    CLIPTextModel, CLIPTokenizer,
    BertModel, BertTokenizer,
    T5EncoderModel, T5Tokenizer,
    GPT2Model, GPT2Tokenizer
)
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import json
import os
from tqdm import tqdm
import re
from collections import Counter


class AdvancedTextPreprocessor:
    """
    Advanced text preprocessing for text-to-image models
    Handles cleaning, normalization, and quality enhancement
    """
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_numbers: bool = False,
                 max_length: int = 77):
        """
        Initialize preprocessor
        
        Args:
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_numbers: Remove numbers
            max_length: Maximum text length
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.max_length = max_length
        
        print(" Text Preprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.,!?\-]', '', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation if requested
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers if requested
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def enhance_prompt(self, text: str, add_quality_tags: bool = True) -> str:
        """
        Enhance prompt with quality tags
        
        Args:
            text: Input text
            add_quality_tags: Add quality enhancement tags
            
        Returns:
            Enhanced text
        """
        if not add_quality_tags:
            return text
        
        # Quality tags for better generation
        quality_tags = [
            "high quality",
            "detailed",
            "professional"
        ]
        
        # Check if already has quality indicators
        has_quality = any(tag in text.lower() for tag in quality_tags)
        
        if not has_quality:
            # Add quality tag at end
            text = f"{text}, high quality, detailed"
        
        return text
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text and return statistics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        words = text.split()
        
        return {
            'length': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'unique_words': len(set(words)),
            'has_numbers': bool(re.search(r'\d', text)),
            'has_punctuation': bool(re.search(r'[^\w\s]', text))
        }
    
    def preprocess(self, text: str, enhance: bool = False) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            enhance: Apply enhancement
            
        Returns:
            Preprocessed text
        """
        # Clean
        text = self.clean_text(text)
        
        # Enhance if requested
        if enhance:
            text = self.enhance_prompt(text)
        
        # Truncate if too long
        if len(text) > self.max_length * 5:  # Approximate
            text = text[:self.max_length * 5]
        
        return text


class UniversalTextEncoder:
    """
    Universal text encoder supporting multiple models from Hugging Face
    Supports: CLIP, BERT, T5, GPT-2
    """
    
    def __init__(self,
                 model_name: str = 'clip',
                 model_variant: str = 'openai/clip-vit-base-patch32',
                 device: str = 'auto',
                 max_length: int = 77,
                 cache_dir: Optional[str] = None):
        """
        Initialize universal text encoder
        
        Args:
            model_name: 'clip', 'bert', 't5', or 'gpt2'
            model_variant: Specific model variant
            device: Device to use
            max_length: Maximum sequence length
            cache_dir: Directory for caching embeddings
        """
        self.model_name = model_name.lower()
        self.model_variant = model_variant
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"\n Initializing {model_name.upper()} Text Encoder")
        print(f"   Model: {model_variant}")
        print(f"   Device: {self.device}")
        print(f"   Max length: {max_length}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        print(f" Encoder ready!")
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def _load_model(self):
        """Load the appropriate model and tokenizer"""
        
        if self.model_name == 'clip':
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_variant)
            self.model = CLIPTextModel.from_pretrained(self.model_variant)
            self.embedding_dim = self.model.config.hidden_size
            
        elif self.model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.model_variant)
            self.model = BertModel.from_pretrained(self.model_variant)
            self.embedding_dim = self.model.config.hidden_size
            
        elif self.model_name == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_variant)
            self.model = T5EncoderModel.from_pretrained(self.model_variant)
            self.embedding_dim = self.model.config.d_model
            
        elif self.model_name == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_variant)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2Model.from_pretrained(self.model_variant)
            self.embedding_dim = self.model.config.hidden_size
        
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Move to device and eval mode
        self.model.to(self.device)
        self.model.eval()
    
    def tokenize(self, 
                 texts: Union[str, List[str]], 
                 return_attention_mask: bool = True) -> Dict:
        """
        Tokenize text(s)
        
        Args:
            texts: Single text or list of texts
            return_attention_mask: Return attention mask
            
        Returns:
            Dictionary with input_ids and optionally attention_mask
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=return_attention_mask
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded.get('attention_mask', None)
        }
    
    def encode(self,
               texts: Union[str, List[str]],
               normalize: bool = True,
               pooling: str = 'mean',
               use_cache: bool = False) -> torch.Tensor:
        """
        Encode text(s) to embeddings
        
        Args:
            texts: Single text or list of texts
            normalize: Normalize embeddings
            pooling: Pooling strategy ('mean', 'max', 'cls', 'pooler')
            use_cache: Use cached embeddings if available
            
        Returns:
            Tensor of embeddings [batch_size, embedding_dim]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache
        if use_cache and all(text in self.embedding_cache for text in texts):
            embeddings = torch.stack([self.embedding_cache[text] for text in texts])
            return embeddings.to(self.device)
        
        # Tokenize
        encoded = self.tokenize(texts)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Encode
        with torch.no_grad():
            if self.model_name == 'clip':
                outputs = self.model(input_ids, attention_mask=attention_mask)
                # CLIP has pooler_output
                embeddings = outputs.pooler_output
                
            elif self.model_name == 'bert':
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                if pooling == 'pooler':
                    embeddings = outputs.pooler_output
                elif pooling == 'cls':
                    embeddings = outputs.last_hidden_state[:, 0, :]
                elif pooling == 'mean':
                    embeddings = self._mean_pooling(
                        outputs.last_hidden_state, attention_mask
                    )
                elif pooling == 'max':
                    embeddings = self._max_pooling(
                        outputs.last_hidden_state, attention_mask
                    )
                    
            elif self.model_name == 't5':
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                if pooling == 'mean':
                    embeddings = self._mean_pooling(
                        outputs.last_hidden_state, attention_mask
                    )
                else:
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    
            elif self.model_name == 'gpt2':
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Use last token for GPT-2
                if pooling == 'mean':
                    embeddings = self._mean_pooling(
                        outputs.last_hidden_state, attention_mask
                    )
                else:
                    # Last non-padding token
                    seq_lengths = attention_mask.sum(dim=1) - 1
                    embeddings = outputs.last_hidden_state[
                        torch.arange(len(seq_lengths)), seq_lengths
                    ]
        
        # Normalize
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Cache
        if use_cache:
            for i, text in enumerate(texts):
                self.embedding_cache[text] = embeddings[i].cpu()
        
        return embeddings
    
    def _mean_pooling(self, hidden_states, attention_mask):
        """Mean pooling over sequence"""
        # Expand mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum and average
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def _max_pooling(self, hidden_states, attention_mask):
        """Max pooling over sequence"""
        # Set padded positions to large negative value
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden_states[mask_expanded == 0] = -1e9
        
        return torch.max(hidden_states, dim=1)[0]
    
    def encode_batch(self,
                     texts: List[str],
                     batch_size: int = 32,
                     show_progress: bool = True) -> torch.Tensor:
        """
        Encode large batch of texts efficiently
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            Tensor of all embeddings
        """
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            embeddings = self.encode(batch_texts)
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    def save_cache(self, filepath: str):
        """Save embedding cache to disk"""
        if self.embedding_cache:
            cache_data = {
                text: emb.numpy().tolist()
                for text, emb in self.embedding_cache.items()
            }
            with open(filepath, 'w') as f:
                json.dump(cache_data, f)
            print(f"💾 Saved {len(cache_data)} cached embeddings to {filepath}")
    
    def load_cache(self, filepath: str):
        """Load embedding cache from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                cache_data = json.load(f)
            
            self.embedding_cache = {
                text: torch.tensor(emb)
                for text, emb in cache_data.items()
            }
            print(f" Loaded {len(cache_data)} cached embeddings from {filepath}")


class TextEncoderPipeline:
    """
    Complete text encoding pipeline
    Combines preprocessing and encoding
    """
    
    def __init__(self,
                 encoder_model: str = 'clip',
                 encoder_variant: str = 'openai/clip-vit-base-patch32',
                 preprocess: bool = True,
                 enhance_prompts: bool = False,
                 device: str = 'auto'):
        """
        Initialize complete pipeline
        
        Args:
            encoder_model: Encoder model name
            encoder_variant: Specific variant
            preprocess: Enable preprocessing
            enhance_prompts: Enable prompt enhancement
            device: Device to use
        """
        print("=" * 80)
        print("TEXT ENCODER PIPELINE")
        print("=" * 80)
        
        # Initialize preprocessor
        if preprocess:
            self.preprocessor = AdvancedTextPreprocessor()
        else:
            self.preprocessor = None
        
        self.enhance_prompts = enhance_prompts
        
        # Initialize encoder
        self.encoder = UniversalTextEncoder(
            model_name=encoder_model,
            model_variant=encoder_variant,
            device=device
        )
        
        print("\n Pipeline ready!")
    
    def process_and_encode(self,
                          texts: Union[str, List[str]],
                          return_tokens: bool = False,
                          return_stats: bool = False) -> Dict:
        """
        Complete pipeline: preprocess and encode
        
        Args:
            texts: Input text(s)
            return_tokens: Return tokenized inputs
            return_stats: Return text statistics
            
        Returns:
            Dictionary with embeddings and optional data
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess
        if self.preprocessor:
            processed_texts = [
                self.preprocessor.preprocess(text, enhance=self.enhance_prompts)
                for text in texts
            ]
        else:
            processed_texts = texts
        
        # Encode
        embeddings = self.encoder.encode(processed_texts)
        
        # Prepare output
        result = {
            'embeddings': embeddings,
            'processed_texts': processed_texts
        }
        
        # Add tokens if requested
        if return_tokens:
            tokens = self.encoder.tokenize(processed_texts)
            result['tokens'] = tokens
        
        # Add stats if requested
        if return_stats and self.preprocessor:
            stats = [
                self.preprocessor.analyze_text(text)
                for text in processed_texts
            ]
            result['stats'] = stats
        
        return result
    
    def __call__(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Convenience method for quick encoding
        
        Args:
            texts: Input text(s)
            
        Returns:
            Embeddings tensor
        """
        result = self.process_and_encode(texts)
        return result['embeddings']


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_encoders(text: str, models: List[str] = None):
    """
    Compare different encoder models on the same text
    
    Args:
        text: Input text
        models: List of model names to compare
    """
    if models is None:
        models = ['clip', 'bert']
    
    print(f"\n{'='*80}")
    print(f"COMPARING ENCODERS")
    print(f"{'='*80}")
    print(f"\nInput text: '{text}'")
    print(f"\nModels: {', '.join(models)}")
    
    results = {}
    
    for model_name in models:
        print(f"\n{model_name.upper()}:")
        
        # Select appropriate variant
        variants = {
            'clip': 'openai/clip-vit-base-patch32',
            'bert': 'bert-base-uncased',
            't5': 't5-small',
            'gpt2': 'gpt2'
        }
        
        encoder = UniversalTextEncoder(
            model_name=model_name,
            model_variant=variants[model_name]
        )
        
        embeddings = encoder.encode([text])
        
        print(f"   Embedding shape: {embeddings.shape}")
        print(f"   Embedding dim: {encoder.get_embedding_dim()}")
        print(f"   Norm: {embeddings.norm().item():.4f}")
        
        results[model_name] = {
            'embeddings': embeddings,
            'dim': encoder.get_embedding_dim()
        }
    
    return results


def analyze_dataset_captions(captions: List[str], sample_size: int = 1000):
    """
    Analyze caption statistics from dataset
    
    Args:
        captions: List of captions
        sample_size: Number of samples to analyze
    """
    preprocessor = AdvancedTextPreprocessor()
    
    captions_sample = captions[:sample_size]
    
    print(f"\n{'='*80}")
    print(f"CAPTION ANALYSIS")
    print(f"{'='*80}")
    print(f"\nAnalyzing {len(captions_sample):,} captions...")
    
    stats = [preprocessor.analyze_text(cap) for cap in tqdm(captions_sample)]
    
    # Aggregate statistics
    lengths = [s['length'] for s in stats]
    word_counts = [s['word_count'] for s in stats]
    avg_word_lengths = [s['avg_word_length'] for s in stats]
    
    print(f"\n STATISTICS:")
    print(f"   Character Length:")
    print(f"      Min: {min(lengths)}")
    print(f"      Max: {max(lengths)}")
    print(f"      Mean: {np.mean(lengths):.1f}")
    print(f"      Median: {np.median(lengths):.1f}")
    
    print(f"\n   Word Count:")
    print(f"      Min: {min(word_counts)}")
    print(f"      Max: {max(word_counts)}")
    print(f"      Mean: {np.mean(word_counts):.1f}")
    print(f"      Median: {np.median(word_counts):.1f}")
    
    print(f"\n   Average Word Length:")
    print(f"      Mean: {np.mean(avg_word_lengths):.1f}")
    
    # Most common words
    all_words = []
    for cap in captions_sample:
        all_words.extend(cap.lower().split())
    
    word_freq = Counter(all_words)
    
    print(f"\n   Top 10 most common words:")
    for word, count in word_freq.most_common(10):
        print(f"      '{word}': {count}")
    
    return stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("TEXT ENCODING SYSTEM - EXAMPLES")
    print("=" * 80)
    
    # Example 1: Basic encoding
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Text Encoding")
    print("="*80)
    
    encoder = UniversalTextEncoder(model_name='clip')
    
    texts = [
        "a beautiful sunset over the ocean",
        "a cat sitting on a windowsill",
        "a modern cityscape at night"
    ]
    
    embeddings = encoder.encode(texts)
    print(f"\nInput: {len(texts)} texts")
    print(f"Output: {embeddings.shape}")
    print(f"Embedding dimension: {encoder.get_embedding_dim()}")
    
    # Example 2: Complete pipeline
    print("\n" + "="*80)
    print("EXAMPLE 2: Complete Pipeline with Preprocessing")
    print("="*80)
    
    pipeline = TextEncoderPipeline(
        encoder_model='clip',
        preprocess=True,
        enhance_prompts=True
    )
    
    text = "beautiful mountain landscape"
    result = pipeline.process_and_encode(text, return_stats=True)
    
    print(f"\nOriginal: '{text}'")
    print(f"Processed: '{result['processed_texts'][0]}'")
    print(f"Embedding shape: {result['embeddings'].shape}")
    print(f"Stats: {result['stats'][0]}")
    
    # Example 3: Batch processing
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch Processing")
    print("="*80)
    
    large_batch = [f"sample text number {i}" for i in range(100)]
    
    batch_embeddings = encoder.encode_batch(large_batch, batch_size=32)
    print(f"\nProcessed {len(large_batch)} texts")
    print(f"Output shape: {batch_embeddings.shape}")
    
    print("\n" + "="*80)
    print(" ALL EXAMPLES COMPLETE!")
    print("="*80)
