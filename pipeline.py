"""
Complete Text-to-Image Pipeline
Integrates all components: preprocessing, embedding, and generation
Supports both Stable Diffusion and GAN-based approaches
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import os
import sys
from datetime import datetime

# Add paths
sys.path.append('./models')
sys.path.append('./utils')

# Import components
from text_preprocessing import AdvancedTextPreprocessor, PromptTemplateEngine
from text_embedding import CachedTextEmbedder
from text_to_image_gan import TextToImageGAN, weights_init


class TextToImagePipeline:
    """
    Unified pipeline for text-to-image generation
    Supports multiple backends (Stable Diffusion, GAN)
    """
    
    def __init__(
        self,
        backend: str = 'gan',
        device: str = 'auto',
        enable_preprocessing: bool = True,
        enable_caching: bool = True
    ):
        """
        Initialize Text-to-Image Pipeline
        
        Args:
            backend: Generation backend ('gan' or 'stable_diffusion')
            device: Device to use ('auto', 'cuda', 'cpu')
            enable_preprocessing: Enable text preprocessing
            enable_caching: Enable embedding caching
        """
        self.backend = backend
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Initializing Text-to-Image Pipeline")
        print(f"🔧 Backend: {backend}")
        print(f"📍 Device: {self.device}")
        
        # Initialize text preprocessing
        if enable_preprocessing:
            self.preprocessor = AdvancedTextPreprocessor(enhance_prompts=True)
            self.template_engine = PromptTemplateEngine()
            print("✅ Text Preprocessing: Enabled")
        else:
            self.preprocessor = None
            self.template_engine = None
            print("⚠️  Text Preprocessing: Disabled")
        
        # Initialize text embedding
        print("\n📦 Loading Text Embedding Model...")
        if enable_caching:
            self.text_embedder = CachedTextEmbedder(
                model_name='openai/clip-vit-base-patch32',
                device=self.device,
                cache_size=10000
            )
        else:
            from text_embedding import TextEmbedder
            self.text_embedder = TextEmbedder(
                model_name='openai/clip-vit-base-patch32',
                device=self.device
            )
        
        # Initialize generation model
        print(f"\n🎨 Loading {backend.upper()} Model...")
        if backend == 'gan':
            self.model = self._initialize_gan()
        elif backend == 'stable_diffusion':
            self.model = self._initialize_stable_diffusion()
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        print("\n✅ Pipeline initialized successfully!\n")
    
    def _initialize_gan(self):
        """Initialize GAN model"""
        model = TextToImageGAN(
            latent_dim=100,
            text_embedding_dim=self.text_embedder.get_embedding_dim(),
            ngf=64,
            ndf=64,
            num_channels=3,
            image_size=64,
            num_residual_blocks=3
        )
        
        # Apply weight initialization
        model.generator.apply(weights_init)
        model.discriminator.apply(weights_init)
        
        model = model.to(self.device)
        print(f"✓ GAN model initialized")
        return model
    
    def _initialize_stable_diffusion(self):
        """Initialize Stable Diffusion model"""
        try:
            from stable_diffusion_enhanced import EnhancedStableDiffusionGenerator
            model = EnhancedStableDiffusionGenerator(
                device=str(self.device),
                enable_preprocessing=False  # We handle preprocessing separately
            )
            print(f"✓ Stable Diffusion model loaded")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load Stable Diffusion: {e}")
    
    def preprocess_prompt(
        self,
        prompt: str,
        enhance: bool = True,
        check_safety: bool = True
    ) -> Dict:
        """
        Preprocess text prompt
        
        Args:
            prompt: Raw text prompt
            enhance: Apply prompt enhancement
            check_safety: Check for NSFW content
            
        Returns:
            Processed prompt dictionary
        """
        if self.preprocessor is None:
            return {
                "status": "success",
                "prompt": prompt,
                "negative_prompt": "",
                "original_prompt": prompt
            }
        
        return self.preprocessor.preprocess_for_stable_diffusion(
            prompt,
            enhance=enhance,
            check_safety=check_safety
        )
    
    def embed_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Convert text to embeddings
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Text embeddings as tensor
        """
        return self.text_embedder.encode_to_tensor(text, normalize=True)
    
    def generate(
        self,
        prompt: str,
        num_samples: int = 1,
        enhance_prompt: bool = True,
        **generation_kwargs
    ) -> List[Tuple[Image.Image, Dict]]:
        """
        Generate images from text prompt
        
        Args:
            prompt: Text description
            num_samples: Number of images to generate
            enhance_prompt: Apply automatic prompt enhancement
            **generation_kwargs: Backend-specific generation parameters
            
        Returns:
            List of (image, metadata) tuples
        """
        # Preprocess prompt
        processed = self.preprocess_prompt(prompt, enhance=enhance_prompt)
        
        if processed["status"] != "success":
            raise ValueError(processed["message"])
        
        final_prompt = processed["prompt"]
        
        print(f"🎨 Generating images for: '{final_prompt}'")
        
        # Generate based on backend
        if self.backend == 'gan':
            return self._generate_with_gan(
                final_prompt,
                num_samples,
                processed
            )
        elif self.backend == 'stable_diffusion':
            return self._generate_with_stable_diffusion(
                final_prompt,
                processed,
                **generation_kwargs
            )
    
    def _generate_with_gan(
        self,
        prompt: str,
        num_samples: int,
        processed_data: Dict
    ) -> List[Tuple[Image.Image, Dict]]:
        """Generate images using GAN"""
        # Get text embedding
        text_embedding = self.embed_text(prompt)
        
        # Generate images
        self.model.eval()
        with torch.no_grad():
            generated_tensors = self.model.generate(text_embedding, num_samples=num_samples)
        
        # Convert to PIL images
        results = []
        for i in range(num_samples):
            # Denormalize from [-1, 1] to [0, 255]
            img_tensor = generated_tensors[i]
            img_tensor = (img_tensor + 1) / 2  # [-1, 1] -> [0, 1]
            img_tensor = img_tensor.clamp(0, 1)
            img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            image = Image.fromarray(img_array)
            
            metadata = {
                "prompt": prompt,
                "backend": "gan",
                "sample_index": i,
                "enhanced": processed_data.get("original_prompt") != prompt,
                "device": str(self.device)
            }
            
            results.append((image, metadata))
        
        return results
    
    def _generate_with_stable_diffusion(
        self,
        prompt: str,
        processed_data: Dict,
        **kwargs
    ) -> List[Tuple[Image.Image, Dict]]:
        """Generate images using Stable Diffusion"""
        # Use the enhanced Stable Diffusion generator
        image, metadata = self.model.generate_image(
            prompt=prompt,
            negative_prompt=processed_data.get("negative_prompt", ""),
            auto_enhance=False,  # Already enhanced
            **kwargs
        )
        
        return [(image, metadata)]
    
    def generate_batch(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> List[List[Tuple[Image.Image, Dict]]]:
        """
        Generate images for multiple prompts
        
        Args:
            prompts: List of text prompts
            **generation_kwargs: Generation parameters
            
        Returns:
            List of results for each prompt
        """
        results = []
        total = len(prompts)
        
        print(f"🎨 Batch Generation: {total} prompts\n")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"[{i}/{total}] {prompt}")
            try:
                result = self.generate(prompt, **generation_kwargs)
                results.append(result)
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append([])
        
        return results
    
    def save_images(
        self,
        results: List[Tuple[Image.Image, Dict]],
        output_dir: str = './outputs/generated',
        prefix: str = 'gen'
    ) -> List[str]:
        """
        Save generated images
        
        Args:
            results: List of (image, metadata) tuples
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, (image, metadata) in enumerate(results):
            filename = f"{prefix}_{timestamp}_{i:03d}.png"
            filepath = os.path.join(output_dir, filename)
            
            image.save(filepath)
            saved_paths.append(filepath)
            
            # Save metadata
            import json
            metadata_path = filepath.replace('.png', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths
    
    def load_gan_checkpoint(self, checkpoint_path: str):
        """Load pre-trained GAN checkpoint"""
        if self.backend != 'gan':
            raise ValueError("Can only load GAN checkpoints with GAN backend")
        
        self.model.load(checkpoint_path, device=str(self.device))
        print(f"✅ Loaded GAN checkpoint from {checkpoint_path}")
    
    def get_embedding_stats(self) -> Dict:
        """Get embedding cache statistics"""
        if hasattr(self.text_embedder, 'get_cache_stats'):
            return self.text_embedder.get_cache_stats()
        return {"cache": "disabled"}


def demo_pipeline():
    """
    Demonstration of the complete pipeline
    """
    print("=" * 70)
    print("TEXT-TO-IMAGE GENERATION PIPELINE DEMO")
    print("=" * 70)
    
    # Initialize pipeline with GAN backend
    print("\n1. Initializing Pipeline with GAN Backend\n")
    pipeline = TextToImagePipeline(
        backend='gan',
        device='auto',
        enable_preprocessing=True,
        enable_caching=True
    )
    
    # Test prompts
    test_prompts = [
        "a beautiful sunset over mountains",
        "a cute cat playing with yarn",
        "cyberpunk city at night with neon lights",
        "realistic portrait of an old wizard",
        "abstract colorful geometric shapes"
    ]
    
    print("\n2. Testing Text Preprocessing\n")
    for prompt in test_prompts[:2]:
        processed = pipeline.preprocess_prompt(prompt)
        print(f"Original: {prompt}")
        print(f"Enhanced: {processed['prompt']}")
        print(f"Negative: {processed['negative_prompt'][:50]}...")
        print()
    
    print("\n3. Testing Text Embedding\n")
    embeddings = pipeline.embed_text(test_prompts[:3])
    print(f"Embedded {len(test_prompts[:3])} prompts")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding device: {embeddings.device}")
    
    print("\n4. Testing Image Generation (GAN)\n")
    # Note: This will generate random images since model is not trained
    print("⚠️  Note: Model not trained, outputs will be random noise")
    
    results = pipeline.generate(
        prompt=test_prompts[0],
        num_samples=2,
        enhance_prompt=True
    )
    
    print(f"Generated {len(results)} images")
    for i, (image, metadata) in enumerate(results):
        print(f"  Image {i+1}: {image.size}, Backend: {metadata['backend']}")
    
    # Save results
    saved_paths = pipeline.save_images(results, output_dir='./outputs/demo')
    print(f"\n💾 Images saved to: {saved_paths[0].rsplit('/', 1)[0]}")
    
    # Cache stats
    print("\n5. Embedding Cache Statistics\n")
    stats = pipeline.get_embedding_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    demo_pipeline()
