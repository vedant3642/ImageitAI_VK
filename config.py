import os

# Model Configuration
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"  # FREE model
CACHE_DIR = "./models"  # Local model storage

# Generation Settings
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.5
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

# Style Presets (Prompt Engineering)
STYLE_PRESETS = {
    "None": "",
    "Photorealistic": "ultra realistic, 8k, highly detailed, DSLR, professional photography, volumetric lighting",
    "Artistic": "oil painting, masterpiece, artstation, brush strokes, artistic",
    "Digital Art": "digital art, concept art, trending on artstation, detailed",
    "Cartoon": "pixar style, cartoon, 3d render, smooth, cute",
    "Anime": "anime style, manga, studio ghibli style, vibrant colors",
    "Cyberpunk": "cyberpunk, neon lights, futuristic, sci-fi, blade runner style",
    "Fantasy": "fantasy art, magical, ethereal, mystical, epic",
    "Sketch": "pencil sketch, hand drawn, black and white, artistic"
}

# Negative Prompts (Filter unwanted elements)
DEFAULT_NEGATIVE = "low quality, blurry, distorted, deformed, ugly, bad anatomy"

# Output Settings
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)