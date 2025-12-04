import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime
import json
from config import *

class ImageGenerator:
    def __init__(self, model_id=DEFAULT_MODEL):
        """Initialize the text-to-image generator"""
        print(f"Loading model: {model_id}")
        
        # Check device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=CACHE_DIR,
            safety_checker=None  # Disable for speed (add your own filter)
        )
        
        # Optimize scheduler for speed
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            # self.pipe.enable_xformers_memory_efficient_attention()
        
        print("✅ Model loaded successfully!")
    
    def generate_images(
        self,
        prompt,
        negative_prompt=DEFAULT_NEGATIVE,
        style="None",
        num_images=1,
        num_steps=DEFAULT_STEPS,
        guidance_scale=DEFAULT_GUIDANCE,
        seed=None,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        progress_callback=None,
        final_width=None,
        final_height=None
    ):
        """Generate images from text prompt"""
        
        # Apply style preset
        if style != "None" and style in STYLE_PRESETS:
            enhanced_prompt = f"{prompt}, {STYLE_PRESETS[style]}"
        else:
            enhanced_prompt = prompt
        
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        def callback_function(step, timestep, latents):
            """Called after each denoising step"""
            if progress_callback:
                # Calculate progress percentage
                progress = (step + 1) / num_steps
                progress_callback(progress, step + 1, num_steps)

        # Generate images
        print(f"Generating {num_images} image(s)...")
        results = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
            callback=callback_function,
            callback_steps=1
        )
        
        images = results.images

        if final_width and final_height and (final_width != width or final_height != height):
            from PIL import Image
            print(f"Resizing images to {final_width}x{final_height}...")
            images = [img.resize((final_width, final_height), Image.LANCZOS) for img in images]
        
        # Add watermark
        watermarked_images = [self.add_watermark(img) for img in images]
        
        # Save with metadata, All generation settings to save as metadata so can create the same image again
        saved_paths = self.save_images(
            watermarked_images,
            prompt=prompt,
            enhanced_prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            style=style,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
            final_width=final_width,
            final_height=final_height
        )
        
        return watermarked_images, saved_paths
    
    def add_watermark(self, image):
        """Add 'AI Generated' watermark"""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # Simple text watermark
        text = "🤖 ImageitAI"
        bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        position = (img.width - text_width - 10, img.height - text_height - 10)
        draw.text(position, text, fill=(255, 255, 255, 128))
        
        return img
    
    def save_images(self, images, **metadata):
        """Save images with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(OUTPUT_DIR, timestamp)
        os.makedirs(session_dir, exist_ok=True)
        
        saved_paths = []
        
        for i, img in enumerate(images):
            # Save image
            img_path = os.path.join(session_dir, f"image_{i+1}.png")
            img.save(img_path, "PNG")
            saved_paths.append(img_path)
        
        # Save metadata
        metadata["timestamp"] = timestamp
        metadata["num_images"] = len(images)
        metadata_path = os.path.join(session_dir, "metadata.json")
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return saved_paths