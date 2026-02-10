# Text-to-Image Generation Pipeline

A comprehensive text-to-image generation system supporting both **Stable Diffusion** and **GAN-based** approaches with advanced text preprocessing and embedding.

## Features

### Core Components

1. **Advanced Text Preprocessing**
   - Automatic prompt enhancement for better quality
   - Style detection (realistic, artistic, anime, 3D, abstract)
   - Safety filtering (NSFW content detection)
   - Prompt templates for structured generation
   - Quality boosters and negative prompt generation

2. **Text Embedding**
   - CLIP-based text encoding (optimal for image generation)
   - Alternative BERT/Sentence-Transformer support
   - Built-in embedding cache for efficiency
   - Normalized embeddings for stable training

3. **GAN-based Generator**
   - Conditional GAN architecture
   - Text-conditioned image synthesis
   - Residual blocks for quality refinement
   - Gradient penalty (WGAN-GP) for stable training

4. **Stable Diffusion Integration**
   - Enhanced wrapper for Stable Diffusion v1.5
   - Automatic memory optimization
   - Multiple scheduler support
   - Batch generation capabilities

5. **Training Pipeline**
   - Comprehensive GAN trainer
   - TensorBoard logging
   - Checkpoint management
   - Validation sampling

## Project Structure

```
text-to-image-gan-pipeline/
├── models/
│   ├── text_to_image_gan.py      # GAN architecture
│   ├── stable_diffusion_enhanced.py  # Enhanced SD wrapper
│   └── trainer.py                 # Training pipeline
├── utils/
│   ├── text_preprocessing.py      # Text preprocessing
│   └── text_embedding.py          # Text embedding models
├── configs/
│   └── config.yaml                # Configuration
├── pipeline.py                    # Main pipeline integration
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd text-to-image-gan-pipeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data (First time only)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

## Usage

### Quick Start - Complete Pipeline

```python
from pipeline import TextToImagePipeline

# Initialize pipeline (GAN backend)
pipeline = TextToImagePipeline(
    backend='gan',  # or 'stable_diffusion'
    device='auto',
    enable_preprocessing=True,
    enable_caching=True
)

# Generate images
results = pipeline.generate(
    prompt="a beautiful sunset over mountains",
    num_samples=4,
    enhance_prompt=True
)

# Save images
pipeline.save_images(results, output_dir='./outputs')
```

### Text Preprocessing Only

```python
from utils.text_preprocessing import AdvancedTextPreprocessor

preprocessor = AdvancedTextPreprocessor(enhance_prompts=True)

# Process single prompt
result = preprocessor.preprocess_for_stable_diffusion(
    prompt="a cute cat",
    enhance=True,
    check_safety=True
)

print(f"Enhanced: {result['prompt']}")
print(f"Negative: {result['negative_prompt']}")
print(f"Style: {result['style']}")
```

### Text Embedding Only

```python
from utils.text_embedding import CachedTextEmbedder

embedder = CachedTextEmbedder(
    model_name='openai/clip-vit-base-patch32',
    device='cuda'
)

# Encode texts
texts = ["a sunset", "a cat", "a city"]
embeddings = embedder.encode(texts)

print(f"Shape: {embeddings.shape}")  # (3, 512)
```

### GAN Training

```python
from models.text_to_image_gan import TextToImageGAN, weights_init
from models.trainer import GANTrainer
from utils.text_embedding import TextEmbedder
from torch.utils.data import DataLoader

# Create model
model = TextToImageGAN(
    latent_dim=100,
    text_embedding_dim=768,
    image_size=64
)
model.generator.apply(weights_init)
model.discriminator.apply(weights_init)

# Create text embedder
text_embedder = TextEmbedder(model_name='openai/clip-vit-base-patch32')

# Create trainer
trainer = GANTrainer(
    model=model,
    text_embedder=text_embedder,
    device='cuda',
    learning_rate=0.0002
)

# Train (assuming you have a dataloader)
trainer.train(
    dataloader=train_loader,
    num_epochs=100,
    val_texts=["a sunset", "a cat"],
    save_interval=10
)
```

### Stable Diffusion Enhanced

```python
from models.stable_diffusion_enhanced import EnhancedStableDiffusionGenerator

# Initialize
generator = EnhancedStableDiffusionGenerator(
    model_id="runwayml/stable-diffusion-v1-5",
    device="auto",
    enable_preprocessing=True
)

# Generate single image
image, metadata = generator.generate_image(
    prompt="a beautiful landscape",
    width=512,
    height=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    scheduler="euler_a",
    auto_enhance=True
)

# Generate variations
variations = generator.generate_variations(
    prompt="a cute cat",
    num_variations=4
)

# Batch generation
prompts = ["sunset", "cat", "city", "forest"]
batch_results = generator.generate_batch(prompts)
```

## Prompt Templates

```python
from utils.text_preprocessing import PromptTemplateEngine

engine = PromptTemplateEngine()

# Portrait
prompt = engine.generate(
    'portrait',
    subject="a young woman",
    style="photorealistic"
)

# Landscape
prompt = engine.generate(
    'landscape',
    scene="mountain valley",
    lighting="golden hour"
)

# Custom template
engine.add_template(
    'scifi',
    "{subject}, futuristic, sci-fi, {quality}"
)
```

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Data settings
data:
  image_size: 64
  batch_size: 32

# Text embedding
text:
  max_length: 77
  embedding_dim: 768
  pretrained_model: 'openai/clip-vit-base-patch32'

# GAN architecture
generator:
  latent_dim: 100
  ngf: 64
  num_residual_blocks: 3

discriminator:
  ndf: 64
  num_layers: 4

# Training
training:
  num_epochs: 100
  learning_rate: 0.0002
  lambda_gp: 10
  n_critic: 5
```

## Model Architecture

### GAN Generator
```
Input: [Noise (100) + Text Embedding (768)]
  ↓
Linear + Reshape → [batch, 1024, 4, 4]
  ↓
ConvTranspose2d → [batch, 512, 8, 8]
  ↓
ConvTranspose2d → [batch, 256, 16, 16]
  ↓
ConvTranspose2d → [batch, 128, 32, 32]
  ↓
Residual Blocks (3x) → [batch, 128, 32, 32]
  ↓
ConvTranspose2d → [batch, 64, 64, 64]
  ↓
Conv2d → [batch, 3, 64, 64]
  ↓
Output: RGB Image
```

### GAN Discriminator
```
Input: [Image (3, 64, 64) + Text Embedding (768)]
  ↓
Conv2d → [batch, 64, 32, 32]
  ↓
Conv2d → [batch, 128, 16, 16]
  ↓
Conv2d → [batch, 256, 8, 8]
  ↓
Conv2d → [batch, 512, 4, 4]
  ↓
Text Conditioning (element-wise multiplication)
  ↓
Conv2d → [batch, 1, 1, 1]
  ↓
Output: Validity Score
```

## Training Tips

1. **Data Preparation**
   - Use paired image-text datasets (MS-COCO, Flickr30k, etc.)
   - Preprocess and cache text embeddings
   - Normalize images to [-1, 1]

2. **Hyperparameters**
   - Start with learning rate 0.0002
   - Use n_critic=5 for WGAN-GP
   - Adjust gradient penalty λ (typically 10)

3. **Monitoring**
   - Watch discriminator accuracy (should be around 50-70%)
   - Generator loss should decrease gradually
   - Generate validation samples regularly

4. **Common Issues**
   - **Mode collapse**: Increase diversity in training data
   - **Training instability**: Reduce learning rate, increase n_critic
   - **Low quality**: Add more residual blocks, train longer

## Evaluation Metrics

- Inception Score (IS)
- Fréchet Inception Distance (FID)
- CLIP Score (text-image alignment)
- Human evaluation

## Advanced Features

### Custom Text Encoder
```python
from utils.text_embedding import CustomTextEncoder

encoder = CustomTextEncoder(
    vocab_size=30000,
    embedding_dim=300,
    hidden_dim=512,
    num_layers=2
)
```

### Gradient Penalty (WGAN-GP)
Automatically applied in trainer when `lambda_gp > 0`

### Multi-GPU Training
```python
model = nn.DataParallel(model)
```

## Integration with Your Stable Diffusion Code

To integrate with your existing Stable Diffusion code:

```python
# Your existing code
from your_module import StableDiffusionGenerator

# Wrap with preprocessing
from utils.text_preprocessing import AdvancedTextPreprocessor

preprocessor = AdvancedTextPreprocessor()
your_generator = StableDiffusionGenerator()

# Enhanced generation
def generate_enhanced(prompt, **kwargs):
    processed = preprocessor.preprocess_for_stable_diffusion(prompt)
    return your_generator.generate_image(
        prompt=processed['prompt'],
        negative_prompt=processed['negative_prompt'],
        **kwargs
    )
```

## References

- Stable Diffusion: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- CLIP: [Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- WGAN-GP: [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

## Acknowledgments

- Hugging Face Diffusers
- OpenAI CLIP
- Stability AI

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
# Use gradient checkpointing
# Enable CPU offloading for Stable Diffusion
pipe.enable_model_cpu_offload()
```

### Slow Generation
```python
# Use xformers
pip install xformers
pipe.enable_xformers_memory_efficient_attention()
```

### Poor Quality
```python
# Increase inference steps (SD)
# Train GAN longer
# Use better text embeddings (CLIP > BERT for images)
```

## Support

For issues and questions, please open an issue on GitHub.
