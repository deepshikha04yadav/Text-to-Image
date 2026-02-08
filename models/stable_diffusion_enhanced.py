"""
Enhanced Stable Diffusion Generator with Advanced Text Preprocessing
Builds upon the existing StableDiffusionGenerator class
"""

import torch
import torch.nn as nn
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler
)
from torch.cuda.amp import autocast
from PIL import Image
from typing import Optional, Tuple, List, Dict
import time
from datetime import datetime
import os
import gc
from importlib.metadata import version
import json

# Import custom preprocessing (will be in utils/)
import sys
sys.path.append('./utils')
from text_preprocessing import AdvancedTextPreprocessor, PromptTemplateEngine


class EnhancedStableDiffusionGenerator:
    """
    Enhanced Stable Diffusion Generator with integrated text preprocessing,
    batch generation, and advanced features
    """
    
    def __init__(self, 
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "auto",
                 enable_preprocessing: bool = True):
        """
        Initialize Enhanced Stable Diffusion Generator
        
        Args:
            model_id: Hugging Face model ID
            device: Device to use ('auto', 'cuda', 'cpu')
            enable_preprocessing: Enable advanced text preprocessing
        """
        try:
            # Setup device
            self.device = self._setup_device(device)
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            
            print(f"🚀 Initializing Enhanced Stable Diffusion on {self.device}")
            print(f"📊 Using precision: {self.dtype}")
            
            # Version info
            torch_version = version("torch")
            diffusers_version = version("diffusers")
            print(f"📦 PyTorch version: {torch_version}")
            print(f"📦 Diffusers version: {diffusers_version}")
            
            # Load pipeline
            self.pipe = self._load_pipeline(model_id)
            
            # Scheduler setup
            self.current_scheduler = "euler_a"
            self.schedulers = {
                "euler_a": ("Euler Ancestral", "Fast, good for creative images"),
                "euler": ("Euler", "Deterministic, consistent results"),
                "ddim": ("DDIM", "Classic, good quality, slower"),
                "dpm_solver": ("DPM Solver", "High quality, efficient"),
                "lms": ("LMS", "Linear multistep, stable")
            }
            
            # Text preprocessing
            self.enable_preprocessing = enable_preprocessing
            if enable_preprocessing:
                self.preprocessor = AdvancedTextPreprocessor(enhance_prompts=True)
                self.template_engine = PromptTemplateEngine()
                print("✅ Text Preprocessing: Enabled")
            else:
                self.preprocessor = None
                self.template_engine = None
            
            # Generation history
            self.generation_history = []
            
            print("✅ Enhanced Stable Diffusion Generator Ready!")
            print(f"📝 Available Schedulers: {list(self.schedulers.keys())}")
            
        except Exception as e:
            print(f"❌ Initialization Error: {str(e)}")
            raise
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🎯 GPU Detected: {torch.cuda.get_device_name(0)}")
                print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                device = "cpu"
                print("💻 Using CPU (GPU not available)")
        return torch.device(device)
    
    def _load_pipeline(self, model_id: str) -> StableDiffusionPipeline:
        """Load and optimize Stable Diffusion pipeline"""
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            print("🔧 Applying Memory Optimizations...")
            
            # Memory optimizations
            pipe.enable_attention_slicing()
            print(" ✓ Attention Slicing: Enabled")
            
            pipe.enable_vae_slicing()
            print(" ✓ VAE Slicing: Enabled")
            
            # Try XFormers
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print(" ✓ XFormers Attention: Enabled")
            except Exception as e:
                print(f" ⚠ XFormers: Not available")
            
            # Device placement
            if self.device.type == "cuda":
                try:
                    pipe = pipe.to(self.device)
                    print(" ✓ Full GPU Loading: Success")
                except RuntimeError:
                    print(" ⚠ GPU Memory Limited: Using CPU Offload")
                    pipe.enable_model_cpu_offload()
            else:
                pipe.enable_sequential_cpu_offload()
                print(" ✓ CPU Sequential Offload: Enabled")
            
            return pipe
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def set_scheduler(self, scheduler_name: str) -> bool:
        """Change the sampling scheduler"""
        if scheduler_name not in self.schedulers:
            print(f"❌ Unknown scheduler: {scheduler_name}")
            return False
        
        if scheduler_name == self.current_scheduler:
            return True
        
        scheduler_map = {
            "euler_a": EulerAncestralDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "ddim": DDIMScheduler,
            "dpm_solver": DPMSolverMultistepScheduler,
            "lms": LMSDiscreteScheduler
        }
        
        try:
            scheduler_class = scheduler_map[scheduler_name]
            self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)
            self.current_scheduler = scheduler_name
            name, desc = self.schedulers[scheduler_name]
            print(f"🔄 Scheduler Changed: {name} ({desc})")
            return True
        except Exception as e:
            print(f"❌ Scheduler Error: {e}")
            return False
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        scheduler: str = "euler_a",
        auto_enhance: bool = True
    ) -> Tuple[Image.Image, dict]:
        """
        Generate image with automatic prompt enhancement
        
        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image
            width: Image width (must be multiple of 8)
            height: Image height (must be multiple of 8)
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt (CFG scale)
            seed: Random seed for reproducibility
            scheduler: Sampling scheduler to use
            auto_enhance: Automatically enhance the prompt
            
        Returns:
            Tuple of (generated_image, metadata_dict)
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Preprocess prompt
        if self.enable_preprocessing and self.preprocessor and auto_enhance:
            processed = self.preprocessor.preprocess_for_stable_diffusion(prompt)
            
            if processed["status"] != "success":
                raise ValueError(processed["message"])
            
            prompt = processed["prompt"]
            if not negative_prompt:
                negative_prompt = processed["negative_prompt"]
            
            print(f"📝 Enhanced Prompt: '{prompt}'")
        
        # Set scheduler
        self.set_scheduler(scheduler)
        
        # Generate seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        
        # Ensure dimensions are multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        print(f"🎨 Generating: '{prompt[:60]}...'")
        print(f"📏 Size: {width}x{height}, Steps: {num_inference_steps}, CFG: {guidance_scale}")
        print(f"🎲 Seed: {seed}, Scheduler: {scheduler}")
        
        start_time = time.time()
        
        try:
            with torch.inference_mode():
                if self.device.type == "cuda" and self.dtype == torch.float16:
                    with autocast(self.device.type):
                        result = self.pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt if negative_prompt else None,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator
                        )
                else:
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt if negative_prompt else None,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    )
            
            generation_time = time.time() - start_time
            
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler": scheduler,
                "seed": seed,
                "generation_time": round(generation_time, 2),
                "device": str(self.device),
                "dtype": str(self.dtype),
                "auto_enhanced": auto_enhance and self.enable_preprocessing
            }
            
            # Add to history
            self.generation_history.append(metadata)
            
            print(f"✅ Generated in {generation_time:.2f}s")
            
            return result.images[0], metadata
            
        except torch.cuda.OutOfMemoryError:
            self._cleanup_memory()
            raise RuntimeError(
                "GPU Out of Memory! Try: reducing image size, fewer steps, "
                "or use CPU mode."
            )
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")
        finally:
            self._cleanup_memory()
    
    def generate_batch(
        self,
        prompts: List[str],
        **generation_params
    ) -> List[Tuple[Image.Image, dict]]:
        """
        Generate multiple images from a list of prompts
        
        Args:
            prompts: List of text prompts
            **generation_params: Parameters for generate_image()
            
        Returns:
            List of (image, metadata) tuples
        """
        results = []
        total = len(prompts)
        
        print(f"🎨 Batch Generation: {total} images")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{total}] Processing: {prompt[:50]}...")
            try:
                image, metadata = self.generate_image(prompt, **generation_params)
                results.append((image, metadata))
            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append((None, {"error": str(e)}))
        
        print(f"\n✅ Batch complete: {sum(1 for r in results if r[0] is not None)}/{total} successful")
        return results
    
    def generate_variations(
        self,
        prompt: str,
        num_variations: int = 4,
        **generation_params
    ) -> List[Tuple[Image.Image, dict]]:
        """
        Generate multiple variations of the same prompt with different seeds
        
        Args:
            prompt: Base prompt
            num_variations: Number of variations to generate
            **generation_params: Additional generation parameters
            
        Returns:
            List of (image, metadata) tuples
        """
        print(f"🎨 Generating {num_variations} variations of: '{prompt}'")
        
        results = []
        for i in range(num_variations):
            print(f"\n[{i+1}/{num_variations}]")
            image, metadata = self.generate_image(prompt, **generation_params)
            results.append((image, metadata))
        
        return results
    
    def save_image(self, image: Image.Image, metadata: dict, output_dir: str = "outputs") -> str:
        """Save generated image with metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sd_gen_{timestamp}_s{metadata['seed']}_{metadata['width']}x{metadata['height']}.png"
        filepath = os.path.join(output_dir, filename)
        
        image.save(filepath)
        
        # Save metadata
        metadata_file = filepath.replace('.png', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved: {filepath}")
        return filepath
    
    def save_batch(self, results: List[Tuple[Image.Image, dict]], output_dir: str = "outputs") -> List[str]:
        """Save batch of generated images"""
        saved_paths = []
        for image, metadata in results:
            if image is not None:
                path = self.save_image(image, metadata, output_dir)
                saved_paths.append(path)
        return saved_paths
    
    def _cleanup_memory(self):
        """Clean up memory"""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage"""
        if self.device.type == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        return {"device": "cpu", "note": "CPU memory tracking not available"}
    
    def get_generation_stats(self) -> dict:
        """Get statistics from generation history"""
        if not self.generation_history:
            return {"total_generations": 0}
        
        times = [h["generation_time"] for h in self.generation_history]
        return {
            "total_generations": len(self.generation_history),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times)
        }


if __name__ == '__main__':
    # Example usage
    print("Testing Enhanced Stable Diffusion Generator\n")
    
    # Initialize generator
    generator = EnhancedStableDiffusionGenerator(
        device="auto",
        enable_preprocessing=True
    )
    
    # Test single generation
    test_prompt = "a beautiful sunset over mountains"
    
    image, metadata = generator.generate_image(
        prompt=test_prompt,
        width=512,
        height=512,
        num_inference_steps=20,
        guidance_scale=7.5,
        auto_enhance=True
    )
    
    # Save image
    generator.save_image(image, metadata, "outputs/test")
    
    # Print stats
    print("\n📊 Generation Statistics:")
    stats = generator.get_generation_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
