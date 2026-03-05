"""
SIMPLE ART GENERATOR
Load your trained model and generate images easily
"""

import torch
from PIL import Image
import os
import sys

# Add paths
sys.path.append('./models')

# ============================================================================
# ART GENERATOR CLASS
# ============================================================================

class ArtGenerator:
    """
    Simple wrapper for generating art with your trained model
    """
    
    def __init__(self, checkpoint_path='./outputs/art_gan/checkpoints/model_epoch_100.pt', device='auto'):
        """
        Initialize generator
        
        Args:
            checkpoint_path: Path to your trained model checkpoint
            device: 'cuda', 'cpu', or 'auto' (auto-detect)
        """
        print(" Initializing Art Generator...")
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"   Device: {self.device}")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"\n❌ Checkpoint not found: {checkpoint_path}")
            print("\nAvailable checkpoints:")
            ckpt_dir = os.path.dirname(checkpoint_path)
            if os.path.exists(ckpt_dir):
                for f in os.listdir(ckpt_dir):
                    if f.endswith('.pt'):
                        print(f"   - {os.path.join(ckpt_dir, f)}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Load embedder
        self.embedder = self._load_embedder()
        
        print(" Generator ready!\n")
    
    def _load_model(self, checkpoint_path):
        """Load the GAN model"""
        from attention_gan import AttentionTextToImageGAN
        
        print(f"   Loading model from {os.path.basename(checkpoint_path)}...")
        
        # Create model with SAME config as training
        model = AttentionTextToImageGAN(
            latent_dim=100,
            text_embedding_dim=512,  # Match training!
            ngf=64,
            ndf=64,
            num_channels=3,
            image_size=64,
            num_attention_blocks=2,
            attention_heads=8
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Move to device and eval mode
        model = model.to(self.device)
        model.eval()
        
        epoch = checkpoint.get('epoch', '?')
        print(f"   Model trained for {epoch} epochs")
        
        return model
    
    def _load_embedder(self):
        """Load CLIP text embedder"""
        from transformers import CLIPTextModel, CLIPTokenizer
        
        print("   Loading CLIP text embedder...")
        
        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        text_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
        text_model.to(self.device)
        text_model.eval()
        
        class SimpleEmbedder:
            def __init__(self, tokenizer, model, device):
                self.tokenizer = tokenizer
                self.model = model
                self.device = device
            
            def encode_to_tensor(self, texts, normalize=True):
                if isinstance(texts, str):
                    texts = [texts]
                
                with torch.no_grad():
                    encoded = self.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=77,
                        return_tensors='pt'
                    )
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    outputs = self.model(**encoded)
                    embeddings = outputs.pooler_output
                    
                    if normalize:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                return embeddings
        
        return SimpleEmbedder(tokenizer, text_model, self.device)
    
    def generate(self, prompt, num_samples=1, seed=None):
        """
        Generate images from text prompt
        
        Args:
            prompt: Text description (e.g., "a painting of mountains")
            num_samples: Number of images to generate
            seed: Random seed for reproducibility (optional)
            
        Returns:
            List of PIL Images
        """
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(seed)
        
        # Get text embedding
        text_emb = self.embedder.encode_to_tensor(prompt)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(text_emb, num_samples=num_samples)
        
        # Convert to PIL images
        images = []
        for i in range(len(generated)):
            img_tensor = (generated[i] + 1) / 2  # [-1, 1] -> [0, 1]
            img_tensor = img_tensor.clamp(0, 1)
            img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            images.append(Image.fromarray(img_array))
        
        return images
    
    def save_images(self, images, output_dir='./generated_art', prefix='art'):
        """
        Save generated images
        
        Args:
            images: List of PIL Images
            output_dir: Directory to save in
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for i, img in enumerate(images):
            # Find next available number
            counter = 0
            while True:
                filepath = os.path.join(output_dir, f'{prefix}_{counter:04d}.png')
                if not os.path.exists(filepath):
                    break
                counter += 1
            
            img.save(filepath)
            saved_paths.append(filepath)
            print(f" Saved: {filepath}")
        
        return saved_paths


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """Run in interactive mode"""
    print("=" * 80)
    print("INTERACTIVE ART GENERATOR")
    print("=" * 80)
    
    # Find checkpoint
    default_checkpoint = './outputs/art_gan/checkpoints/model_epoch_100.pt'
    
    if not os.path.exists(default_checkpoint):
        # Look for any checkpoint
        ckpt_dir = './outputs/art_gan/checkpoints'
        if os.path.exists(ckpt_dir):
            checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
            if checkpoints:
                default_checkpoint = os.path.join(ckpt_dir, sorted(checkpoints)[-1])
                print(f"\nUsing checkpoint: {default_checkpoint}")
    
    # Initialize generator
    try:
        generator = ArtGenerator(checkpoint_path=default_checkpoint)
    except FileNotFoundError:
        print("\n No trained model found!")
        print("Please train a model first using TRAIN_FIXED_DIMENSIONS.py")
        return
    
    print("\n" + "=" * 80)
    print("READY TO GENERATE!")
    print("=" * 80)
    print("\nExamples:")
    print("  - a beautiful oil painting of mountains")
    print("  - a detailed pencil drawing of a portrait")
    print("  - a marble sculpture of a human figure")
    print("  - an engraving of classical architecture")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit")
    print("  'save' - Save last generated image")
    print("=" * 80)
    
    last_images = None
    
    while True:
        print("\n" + "-" * 80)
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print(" Goodbye!")
            break
        
        if prompt.lower() == 'save' and last_images:
            output = input("Output filename (default: art.png): ").strip()
            if not output:
                output = 'art.png'
            last_images[0].save(output)
            print(f" Saved to {output}")
            continue
        
        if not prompt:
            continue
        
        # Ask for number of samples
        try:
            num = input("Number of samples (1-4, default 1): ").strip()
            num_samples = int(num) if num else 1
            num_samples = max(1, min(4, num_samples))
        except:
            num_samples = 1
        
        print(f"\n Generating {num_samples} image(s)...")
        
        try:
            images = generator.generate(prompt, num_samples=num_samples)
            last_images = images
            
            # Display info
            print(f" Generated {len(images)} image(s)!")
            print(f"   Prompt: '{prompt}'")
            print(f"   Size: {images[0].size}")
            
            # Save
            save = input("\nSave images? (y/n): ").strip().lower()
            if save == 'y':
                saved = generator.save_images(images, prefix=prompt[:30].replace(' ', '_'))
                print(f" Saved {len(saved)} images")
            
            # Display (if in notebook)
            try:
                from IPython.display import display
                for img in images:
                    display(img)
            except:
                print("   (Images saved but cannot display - not in notebook)")
        
        except Exception as e:
            print(f" Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate art with your trained model')
    parser.add_argument('--checkpoint', type=str, 
                       default='./outputs/art_gan/checkpoints/model_epoch_100.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--num-samples', type=int, default=1, 
                       help='Number of images to generate')
    parser.add_argument('--output', type=str, default='./generated_art',
                       help='Output directory')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive or not args.prompt:
        interactive_mode()
    else:
        # Single generation
        generator = ArtGenerator(checkpoint_path=args.checkpoint)
        
        print(f"\n Generating: '{args.prompt}'")
        images = generator.generate(args.prompt, 
                                    num_samples=args.num_samples,
                                    seed=args.seed)
        
        saved = generator.save_images(images, output_dir=args.output)
        
        print(f"\n Generated and saved {len(saved)} images!")
        for path in saved:
            print(f"   {path}")
