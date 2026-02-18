"""
Automatic Caption Generation for Images
Uses BLIP-2, BLIP, or CLIP Interrogator
"""

import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration
)
from typing import List, Optional
import os
from tqdm import tqdm
import json


class AutoCaptioner:
    """
    Automatically generate captions for images
    """
    
    def __init__(
        self,
        model_type: str = "blip2",
        device: str = "auto"
    ):
        """
        Initialize caption generator
        
        Args:
            model_type: 'blip', 'blip2', or 'clip_interrogator'
            device: Device to use
        """
        self.model_type = model_type
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Initializing {model_type.upper()} Caption Generator")
        print(f"📍 Device: {self.device}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the captioning model"""
        if self.model_type == "blip2":
            print("📦 Loading BLIP-2 (best quality, slower)...")
            self.processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b"
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
        elif self.model_type == "blip":
            print("📦 Loading BLIP (good quality, faster)...")
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
    
    def caption_image(
        self,
        image_path: str,
        max_length: int = 50,
        num_beams: int = 5
    ) -> str:
        """
        Generate caption for a single image
        
        Args:
            image_path: Path to image
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            Generated caption
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        early_stopping=True
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
        
        # Decode caption
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption
    
    def caption_directory(
        self,
        images_dir: str,
        output_dir: Optional[str] = None,
        save_json: bool = True,
        save_txt: bool = True,
        batch_size: int = 1
    ) -> dict:
        """
        Generate captions for all images in a directory
        
        Args:
            images_dir: Directory containing images
            output_dir: Directory to save captions (default: same as images_dir)
            save_json: Save all captions in a JSON file
            save_txt: Save individual .txt files for each image
            batch_size: Processing batch size
            
        Returns:
            Dictionary of {image_name: caption}
        """
        if output_dir is None:
            output_dir = images_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        
        print(f"\n📊 Found {len(image_files)} images")
        print(f"🔄 Generating captions...\n")
        
        captions = {}
        
        # Process images
        for img_file in tqdm(image_files, desc="Captioning"):
            img_path = os.path.join(images_dir, img_file)
            
            try:
                # Generate caption
                caption = self.caption_image(img_path)
                captions[img_file] = caption
                
                # Save individual .txt file
                if save_txt:
                    base_name = os.path.splitext(img_file)[0]
                    txt_path = os.path.join(output_dir, f"{base_name}.txt")
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
            
            except Exception as e:
                print(f"❌ Error processing {img_file}: {e}")
                captions[img_file] = ""
        
        # Save JSON file
        if save_json:
            json_path = os.path.join(output_dir, 'captions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(captions, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved captions to {json_path}")
        
        print(f"✅ Generated {len(captions)} captions")
        
        return captions


class CLIPInterrogator:
    """
    Generate prompt-style captions using CLIP Interrogator
    Best for artistic images
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize CLIP Interrogator"""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print("🚀 Initializing CLIP Interrogator")
        print("⚠️  Note: Requires clip-interrogator package")
        print("   Install: pip install clip-interrogator")
        
        try:
            from clip_interrogator import Config, Interrogator
            
            config = Config(device=str(self.device))
            self.interrogator = Interrogator(config)
            print("✅ CLIP Interrogator loaded!")
            
        except ImportError:
            print("❌ clip-interrogator not installed")
            print("   Install with: pip install clip-interrogator")
            self.interrogator = None
    
    def interrogate(self, image_path: str, mode: str = "best") -> str:
        """
        Generate prompt-style caption
        
        Args:
            image_path: Path to image
            mode: 'best', 'fast', or 'classic'
            
        Returns:
            Prompt-style caption
        """
        if self.interrogator is None:
            raise RuntimeError("CLIP Interrogator not available")
        
        image = Image.open(image_path).convert('RGB')
        
        if mode == "best":
            prompt = self.interrogator.interrogate(image)
        elif mode == "fast":
            prompt = self.interrogator.interrogate_fast(image)
        else:
            prompt = self.interrogator.interrogate_classic(image)
        
        return prompt


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def prepare_dataset_structure(
    source_dir: str,
    output_dir: str,
    train_split: float = 0.9
):
    """
    Organize images into train/val structure
    
    Args:
        source_dir: Directory with all images
        output_dir: Output directory for organized dataset
        train_split: Fraction for training set
    """
    import shutil
    import random
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images
    image_files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ]
    
    # Shuffle
    random.shuffle(image_files)
    
    # Split
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"📊 Dataset Split:")
    print(f"   Training: {len(train_files)} images")
    print(f"   Validation: {len(val_files)} images")
    
    # Create directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Copy files
    print("\n🔄 Organizing files...")
    
    for img_file in tqdm(train_files, desc="Training set"):
        src = os.path.join(source_dir, img_file)
        dst = os.path.join(train_dir, img_file)
        shutil.copy2(src, dst)
        
        # Copy caption if exists
        base_name = os.path.splitext(img_file)[0]
        caption_file = f"{base_name}.txt"
        caption_src = os.path.join(source_dir, caption_file)
        if os.path.exists(caption_src):
            caption_dst = os.path.join(train_dir, caption_file)
            shutil.copy2(caption_src, caption_dst)
    
    for img_file in tqdm(val_files, desc="Validation set"):
        src = os.path.join(source_dir, img_file)
        dst = os.path.join(val_dir, img_file)
        shutil.copy2(src, dst)
        
        # Copy caption if exists
        base_name = os.path.splitext(img_file)[0]
        caption_file = f"{base_name}.txt"
        caption_src = os.path.join(source_dir, caption_file)
        if os.path.exists(caption_src):
            caption_dst = os.path.join(val_dir, caption_file)
            shutil.copy2(caption_src, caption_dst)
    
    print(f"\n✅ Dataset organized in {output_dir}")
    print(f"   {train_dir}")
    print(f"   {val_dir}")


if __name__ == '__main__':
    print("=" * 80)
    print("AUTOMATIC CAPTION GENERATOR")
    print("=" * 80)
    
    print("""
Usage Examples:

1. Caption all images in a directory:
   ```python
   captioner = AutoCaptioner(model_type='blip2')
   captioner.caption_directory(
       images_dir='./data/my_images',
       output_dir='./data/my_images_captioned'
   )
   ```

2. Caption single image:
   ```python
   captioner = AutoCaptioner(model_type='blip')
   caption = captioner.caption_image('path/to/image.jpg')
   print(caption)
   ```

3. Use CLIP Interrogator for artistic images:
   ```python
   interrogator = CLIPInterrogator()
   prompt = interrogator.interrogate('artwork.jpg', mode='best')
   print(prompt)
   ```

4. Organize dataset:
   ```python
   prepare_dataset_structure(
       source_dir='./raw_images',
       output_dir='./organized_dataset',
       train_split=0.9
   )
   ```
    """)
