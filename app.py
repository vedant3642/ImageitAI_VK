import streamlit as st
from generator import ImageGenerator
from config import STYLE_PRESETS
import time

# Page config
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Title
st.title("🤖 ImageitAI : AI-Powered Text-to-Image Generator")
st.markdown("Generate high-quality images from text descriptions using AI")

# Initialize generator (cached)
@st.cache_resource
def load_generator():
    return ImageGenerator()

with st.spinner("Loading AI model... (first time may take a few minutes)"):
    generator = load_generator()

st.success("✅ Model loaded! Ready to generate images.")

# Sidebar - Settings
st.sidebar.header("⚙️ Generation Settings")

num_images = st.sidebar.slider("Number of Images", 1, 4, 1)
style = st.sidebar.selectbox("Style Preset", list(STYLE_PRESETS.keys()))
num_steps = st.sidebar.slider("Quality Steps", 20, 50, 30, help="Higher = Better quality but slower")
guidance_scale = st.sidebar.slider("Prompt Guidance", 5.0, 15.0, 7.5, 0.5, help="Higher = More adherence to prompt")
seed = st.sidebar.number_input("Seed (for reproducibility)", value=None, placeholder="Random")

# Advanced settings
with st.sidebar.expander("🔧 Advanced"):
    width = st.selectbox("Width", [512, 768, 1024], index=0)
    height = st.selectbox("Height", [512, 768, 1024], index=0)

    #If you want to resize the size of output images
    resize_output = st.checkbox("Resize output images?", value=False)
    if resize_output:
        final_width = st.number_input("Final Width", min_value=128, max_value=2048, value=256)
        final_height = st.number_input("Final Height", min_value=128, max_value=2048, value=256)

    negative_prompt = st.text_area(
        "Negative Prompt",
        value="low quality, blurry, distorted, difigured, bad anatomy",
        help="Things to avoid in the image"
    )

# Main input
prompt = st.text_area(
    "Enter your text prompt",
    placeholder="Example: a futuristic city at sunset, highly detailed, 8k",
    height=100
)

# Example prompts
with st.expander("💡 Example Prompts"):
    st.markdown("""
    - `a majestic lion in a savanna at golden hour, professional photography`
    - `portrait of a robot in Van Gogh style, oil painting`
    - `a cozy coffee shop in Tokyo, anime style, warm lighting`
    - `futuristic cyberpunk city with neon lights, blade runner style`
    - `mystical forest with glowing mushrooms, fantasy art, magical`
    """)

# Generate button
if st.button("🚀 Generate Images", type="primary"):
    if not prompt:
        st.error("Please enter a prompt!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, current_step, total_steps):
            """Callback function to update progress bar"""
            progress_bar.progress(progress)
            status_text.text(f"🎨 Generating... Step {current_step}/{total_steps} ({int(progress*100)}%)")
        
        start_time = time.time()
        
        # Generate with progress callback
        images, paths = generator.generate_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            style=style,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed if seed else None,
            width=width,
            height=height,
            final_width=final_width if resize_output else None,
            final_height=final_height if resize_output else None,
            progress_callback=update_progress  #callback function
        )
        
        elapsed = time.time() - start_time
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Generated {len(images)} image(s) in {elapsed:.1f} seconds!")
        
        # Display images
        cols = st.columns(min(num_images, 3))
        for i, (img, path) in enumerate(zip(images, paths)):
            with cols[i % 3]:
                st.image(img, caption=f"Image {i+1}", use_container_width=True)
                
                # Download button
                with open(path, "rb") as f:
                    st.download_button(
                        label="📥 Download",
                        data=f.read(),
                        file_name=f"generated_image_{i+1}.png",
                        mime="image/png",
                        key=f"download_{i}"
                    )
        
        # Show metadata
        with st.expander("📊 Generation Details"):
            st.json({
                "prompt": prompt,
                "style": style,
                "steps": num_steps,
                "guidance": guidance_scale,
                "seed": seed,
                "time": f"{elapsed:.2f}s",
                "saved_to": paths[0].rsplit('/', 1)[0]
            })
# Footer
st.markdown("---")
st.markdown("⚠️ **Responsible AI Use**: Generated images are watermarked. Use ethically and respect copyright.")