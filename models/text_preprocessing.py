"""
Advanced Text Preprocessing Module for Text-to-Image Generation
Integrates with both Stable Diffusion and GAN-based pipelines
"""

import re
import string
import nltk
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter

class AdvancedTextPreprocessor:
    """
    Advanced text preprocessing for text-to-image generation
    Handles prompt engineering, quality enhancement, and normalization
    """
    
    def __init__(self, 
                 enhance_prompts: bool = True,
                 max_length: int = 77,  # CLIP's max token length
                 remove_nsfw: bool = True):
        """
        Initialize advanced text preprocessor
        
        Args:
            enhance_prompts: Automatically enhance prompts for better image quality
            max_length: Maximum token length (77 for CLIP)
            remove_nsfw: Filter out potentially problematic content
        """
        self.enhance_prompts = enhance_prompts
        self.max_length = max_length
        self.remove_nsfw = remove_nsfw
        
        # Download required NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
        
        # Quality enhancement keywords
        self.quality_boosters = [
            "high quality", "detailed", "professional", "sharp focus",
            "8k", "masterpiece", "best quality", "highly detailed"
        ]
        
        # Style modifiers
        self.style_keywords = {
            "realistic": ["photorealistic", "hyperrealistic", "realistic", "photograph"],
            "artistic": ["artwork", "painting", "illustration", "digital art"],
            "3d": ["3d render", "cinema4d", "octane render", "unreal engine"],
            "anime": ["anime", "manga", "anime style", "cel shaded"],
            "abstract": ["abstract", "surreal", "conceptual"]
        }
        
        # NSFW keywords to filter
        self.nsfw_keywords = [
            "nsfw", "nude", "naked", "explicit", "sexual", "xxx",
            "porn", "erotic", "adult", "violence", "gore", "blood"
        ]
    
    def clean_prompt(self, prompt: str) -> str:
        """
        Clean and normalize prompt text
        
        Args:
            prompt: Raw prompt text
            
        Returns:
            Cleaned prompt
        """
        if not isinstance(prompt, str):
            prompt = str(prompt)
        
        # Remove extra whitespace
        prompt = ' '.join(prompt.split())
        
        # Remove special characters that might interfere
        prompt = re.sub(r'[^\w\s,.\-()]', '', prompt)
        
        # Fix common typos
        prompt = prompt.replace('  ', ' ')
        prompt = prompt.strip()
        
        return prompt
    
    def check_nsfw(self, prompt: str) -> Tuple[bool, List[str]]:
        """
        Check if prompt contains NSFW content
        
        Args:
            prompt: Prompt text
            
        Returns:
            Tuple of (is_safe, found_keywords)
        """
        prompt_lower = prompt.lower()
        found_keywords = []
        
        for keyword in self.nsfw_keywords:
            if keyword in prompt_lower:
                found_keywords.append(keyword)
        
        is_safe = len(found_keywords) == 0
        return is_safe, found_keywords
    
    def detect_style(self, prompt: str) -> Optional[str]:
        """
        Detect the intended artistic style from prompt
        
        Args:
            prompt: Prompt text
            
        Returns:
            Detected style or None
        """
        prompt_lower = prompt.lower()
        
        for style, keywords in self.style_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    return style
        
        return None
    
    def enhance_prompt(self, prompt: str, style: Optional[str] = None) -> str:
        """
        Enhance prompt with quality boosters and style modifiers
        
        Args:
            prompt: Original prompt
            style: Desired style (auto-detected if None)
            
        Returns:
            Enhanced prompt
        """
        if not self.enhance_prompts:
            return prompt
        
        # Detect style if not provided
        if style is None:
            style = self.detect_style(prompt)
        
        enhanced = prompt
        
        # Add quality boosters if not already present
        prompt_lower = prompt.lower()
        has_quality = any(booster.lower() in prompt_lower for booster in self.quality_boosters)
        
        if not has_quality:
            # Add appropriate quality boosters
            enhanced = f"{enhanced}, highly detailed, high quality, sharp focus"
        
        # Add style-specific enhancements
        if style and style in self.style_keywords:
            style_kw = self.style_keywords[style][0]
            if style_kw not in prompt_lower:
                enhanced = f"{enhanced}, {style_kw}"
        
        return enhanced
    
    def create_negative_prompt(self, custom_negatives: Optional[str] = None) -> str:
        """
        Create comprehensive negative prompt for better quality
        
        Args:
            custom_negatives: Additional negative prompts
            
        Returns:
            Complete negative prompt
        """
        base_negatives = [
            "low quality", "blurry", "bad anatomy", "worst quality",
            "low resolution", "bad proportions", "jpeg artifacts",
            "ugly", "deformed", "distorted", "disfigured"
        ]
        
        if custom_negatives:
            base_negatives.append(custom_negatives)
        
        return ", ".join(base_negatives)
    
    def extract_keywords(self, prompt: str, top_k: int = 10) -> List[Tuple[str, str]]:
        """
        Extract important keywords with their POS tags
        
        Args:
            prompt: Prompt text
            top_k: Number of top keywords to return
            
        Returns:
            List of (word, pos_tag) tuples
        """
        try:
            from nltk import pos_tag, word_tokenize
            tokens = word_tokenize(prompt.lower())
            tagged = pos_tag(tokens)
            
            # Filter important POS tags (nouns, adjectives, verbs)
            important_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG']
            keywords = [(word, pos) for word, pos in tagged if pos in important_pos]
            
            return keywords[:top_k]
        except:
            # Fallback to simple word extraction
            words = prompt.lower().split()
            return [(word, 'UNKNOWN') for word in words[:top_k]]
    
    def split_long_prompt(self, prompt: str, max_tokens: int = 75) -> List[str]:
        """
        Split long prompts into multiple segments for weighted prompting
        
        Args:
            prompt: Long prompt text
            max_tokens: Maximum tokens per segment
            
        Returns:
            List of prompt segments
        """
        words = prompt.split()
        
        if len(words) <= max_tokens:
            return [prompt]
        
        segments = []
        current_segment = []
        
        for word in words:
            current_segment.append(word)
            if len(current_segment) >= max_tokens:
                segments.append(' '.join(current_segment))
                current_segment = []
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
    def preprocess_for_stable_diffusion(self, 
                                       prompt: str,
                                       enhance: bool = True,
                                       check_safety: bool = True) -> Dict[str, any]:
        """
        Complete preprocessing pipeline for Stable Diffusion
        
        Args:
            prompt: Raw prompt
            enhance: Apply prompt enhancement
            check_safety: Check for NSFW content
            
        Returns:
            Dictionary with processed prompt and metadata
        """
        # Clean prompt
        cleaned_prompt = self.clean_prompt(prompt)
        
        # Safety check
        is_safe = True
        nsfw_keywords = []
        if check_safety and self.remove_nsfw:
            is_safe, nsfw_keywords = self.check_nsfw(cleaned_prompt)
            if not is_safe:
                return {
                    "status": "unsafe",
                    "prompt": None,
                    "negative_prompt": None,
                    "nsfw_keywords": nsfw_keywords,
                    "message": f"Prompt contains potentially problematic content: {', '.join(nsfw_keywords)}"
                }
        
        # Enhance prompt
        final_prompt = cleaned_prompt
        if enhance:
            final_prompt = self.enhance_prompt(cleaned_prompt)
        
        # Create negative prompt
        negative_prompt = self.create_negative_prompt()
        
        # Extract metadata
        style = self.detect_style(final_prompt)
        keywords = self.extract_keywords(final_prompt)
        
        return {
            "status": "success",
            "prompt": final_prompt,
            "negative_prompt": negative_prompt,
            "original_prompt": prompt,
            "cleaned_prompt": cleaned_prompt,
            "style": style,
            "keywords": keywords,
            "is_safe": is_safe,
            "token_count": len(final_prompt.split())
        }
    
    def batch_preprocess(self, prompts: List[str]) -> List[Dict]:
        """
        Preprocess multiple prompts
        
        Args:
            prompts: List of raw prompts
            
        Returns:
            List of processed prompt dictionaries
        """
        return [self.preprocess_for_stable_diffusion(prompt) for prompt in prompts]


class PromptTemplateEngine:
    """
    Template engine for creating structured prompts
    """
    
    def __init__(self):
        self.templates = {
            "portrait": "{subject}, portrait, {style}, {quality}",
            "landscape": "{scene}, landscape, {style}, {lighting}, {quality}",
            "object": "{object}, {background}, {style}, {quality}",
            "character": "{character}, {action}, {environment}, {style}, {quality}",
            "abstract": "{concept}, abstract art, {colors}, {mood}, {quality}"
        }
        
        self.defaults = {
            "style": "highly detailed",
            "quality": "high quality, 8k, masterpiece",
            "lighting": "dramatic lighting",
            "background": "simple background",
            "colors": "vibrant colors",
            "mood": "ethereal"
        }
    
    def generate(self, template_type: str, **kwargs) -> str:
        """
        Generate prompt from template
        
        Args:
            template_type: Type of template to use
            **kwargs: Template variables
            
        Returns:
            Generated prompt
        """
        if template_type not in self.templates:
            raise ValueError(f"Unknown template: {template_type}")
        
        template = self.templates[template_type]
        
        # Fill in defaults for missing values
        for key, default_value in self.defaults.items():
            if key not in kwargs:
                kwargs[key] = default_value
        
        try:
            prompt = template.format(**kwargs)
            return prompt
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {e}")
    
    def add_template(self, name: str, template: str):
        """Add custom template"""
        self.templates[name] = template


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("Advanced Text Preprocessing for Text-to-Image Generation")
    print("=" * 60)
    
    preprocessor = AdvancedTextPreprocessor(enhance_prompts=True)
    
    # Test prompts
    test_prompts = [
        "a beautiful sunset",
        "cyberpunk city at night with neon lights",
        "realistic portrait of an old man",
        "cute cat playing with yarn, anime style",
        "abstract geometric shapes"
    ]
    
    print("\n📝 Processing Test Prompts:\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Original: \"{prompt}\"")
        result = preprocessor.preprocess_for_stable_diffusion(prompt)
        
        if result["status"] == "success":
            print(f"   ✓ Enhanced: \"{result['prompt']}\"")
            print(f"   Style: {result['style'] or 'None detected'}")
            print(f"   Keywords: {[kw[0] for kw in result['keywords'][:5]]}")
        else:
            print(f"   ✗ {result['message']}")
    
    # Test template engine
    print("\n\n📋 Template Engine Examples:\n")
    
    template_engine = PromptTemplateEngine()
    
    examples = [
        ("portrait", {"subject": "a young woman", "style": "photorealistic"}),
        ("landscape", {"scene": "mountain valley", "lighting": "golden hour"}),
        ("character", {"character": "knight in armor", "action": "standing heroically", 
                      "environment": "medieval castle", "style": "fantasy art"})
    ]
    
    for template_type, kwargs in examples:
        prompt = template_engine.generate(template_type, **kwargs)
        print(f"{template_type.upper()}: {prompt}")
    
    print("\n" + "=" * 60)
