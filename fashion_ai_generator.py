import torch
import numpy as np
import cv2
import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check GPU availability with detailed info
if torch.cuda.is_available():
    device = "cuda"
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
else:
    device = "cpu"
    logger.info("CUDA not available, using CPU (this will be slow)")

# Load models with better error handling
def load_models():
    logger.info("Starting to load models...")
    model_id = "runwayml/stable-diffusion-v1-5"

    # Create cache directory if it doesn't exist
    os.makedirs(os.path.expanduser("~/.cache/huggingface/diffusers"), exist_ok=True)

    try:
        logger.info(f"Loading text-to-image pipeline from {model_id}...")
        txt2img_pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            cache_dir=os.path.expanduser("~/.cache/huggingface/diffusers")
        )
        txt2img_pipeline = txt2img_pipeline.to(device)
        logger.info("Text-to-image pipeline loaded successfully")

        logger.info(f"Loading image-to-image pipeline from {model_id}...")
        img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            cache_dir=os.path.expanduser("~/.cache/huggingface/diffusers")
        )
        img2img_pipeline = img2img_pipeline.to(device)
        logger.info("Image-to-image pipeline loaded successfully")

        return txt2img_pipeline, img2img_pipeline

    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        logger.error("Creating dummy pipelines for demo purposes")

        class DummyPipeline:
            def __call__(self, **kwargs):
                logger.warning("Using dummy pipeline - models failed to load")
                class DummyOutput:
                    images = [Image.new('RGB', (512, 512), color=(200, 200, 200))]
                return DummyOutput()

        return DummyPipeline(), DummyPipeline()

# Define categories and attributes for Fashion Generator
categories = ["Dress", "T-shirt", "Jeans", "Jacket", "Shirt and Jeans", "Lehenga", "Churidar", "Saree", "Gown"]
styles = ["Modern", "Vintage", "Casual", "Elegant", "Traditional"]
color_schemes = ["Monochrome", "Pastel", "Vibrant"]
patterns = ["Striped", "Floral", "Plain", "Embroidery"]
materials = ["Cotton", "Silk", "Denim", "Leather", "Georgette", "Crepe"]
fits = ["Slim", "Regular", "Loose"]
occasions = ["Formal", "Casual", "Party"]
seasons = ["Summer", "Winter", "Normal"]

# Specific colors for the Similar Image tab
colors = ["Red", "Blue", "Green", "Yellow", "Purple", "Pink", "Orange", "Black", "White", "Brown", "Gray", "Teal"]

# Function for Fashion Generation with improved error handling
def get_completion(input_type, category, style, color_scheme, pattern, material, fit, occasion, season, custom_prompt):
    """Generate fashion image based on selected parameters or custom prompt"""
    try:
        if input_type == "Dropdowns":
            prompt = f"High quality fashion photography of a {fit} {style} {category} made of {material} with {pattern} pattern in {color_scheme} colors, suitable for {occasion} wear during {season} season"
        else:
            prompt = custom_prompt if custom_prompt else "A fashion item"

        logger.info(f"Generating image with prompt: {prompt}")

        # Set guidance scale based on device to improve quality
        guidance_scale = 7.5
        steps = 30 if device == "cuda" else 20  # Reduce steps if on CPU for speed

        # Run pipeline with proper parameters
        result = txt2img_pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt="low quality, blurry, distorted, deformed, disfigured, bad anatomy"  # Improve quality
        )

        if hasattr(result, 'images') and len(result.images) > 0:
            logger.info("Image generated successfully")
            return result.images[0]
        else:
            logger.error("Pipeline returned no images")
            return Image.new('RGB', (512, 512), color=(255, 100, 100))

    except Exception as e:
        logger.error(f"Error in image generation: {str(e)}")
        error_img = Image.new('RGB', (512, 512), color=(255, 100, 100))
        # Add error text to image
        draw = ImageDraw.Draw(error_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((50, 200), f"Error: {str(e)[:50]}...", fill=(0, 0, 0), font=font)
        return error_img

# Function to toggle between dropdown inputs and custom prompt
def toggle_input_type(choice):
    """Toggle visibility between dropdown options and custom text input"""
    if choice == "Dropdowns":
        return [gr.update(visible=True) for _ in range(8)] + [gr.update(visible=False)]
    else:  # "Manual Text"
        return [gr.update(visible=False) for _ in range(8)] + [gr.update(visible=True)]

# Updated Function for Similar Image Generation with color and style options
def generate_similar_image(reference_image, strength=0.7, target_color=None, style_modifier=None, maintain_structure=True, enhance_details=False):
    """Generate an image similar to the uploaded reference image using img2img with additional features"""
    try:
        if reference_image is None:
            logger.warning("No reference image provided")
            return None

        if isinstance(reference_image, np.ndarray):
            reference_image = Image.fromarray(reference_image)

        reference_image = reference_image.convert("RGB").resize((512, 512))

        # Build prompt based on selected options
        prompt = "high quality fashion item, detailed clothing, professional fashion photography"

        # Add color modification if selected
        if target_color and target_color != "Original":
            prompt += f", {target_color} colored"
            logger.info(f"Adding color modification: {target_color}")

        # Add style modification if selected
        if style_modifier and style_modifier != "None":
            prompt += f", {style_modifier} style"
            logger.info(f"Adding style modification: {style_modifier}")

        # Add detail enhancement if selected
        if enhance_details:
            prompt += ", highly detailed, intricate details, high resolution"
            logger.info("Adding detail enhancement")

        logger.info(f"Generating similar image with prompt: {prompt}")
        logger.info(f"Strength: {strength}, Maintain structure: {maintain_structure}")

        # Adjust strength based on maintain structure option
        if maintain_structure and strength > 0.5:
            actual_strength = 0.5  # Lower strength to better maintain structure
        else:
            actual_strength = strength

        # Set guidance scale based on device to improve quality
        guidance_scale = 7.5
        if enhance_details:
            guidance_scale = 8.5  # Increase guidance scale for more detailed output

        steps = 30 if device == "cuda" else 20  # Reduce steps if on CPU for speed

        # Enhanced negative prompt
        negative_prompt = "low quality, blurry, distorted, deformed, disfigured, bad anatomy"

        similar_image = img2img_pipeline(
            prompt=prompt,
            image=reference_image,
            strength=actual_strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            negative_prompt=negative_prompt
        ).images[0]

        return similar_image
    except Exception as e:
        logger.error(f"Error in similar image generation: {str(e)}")
        error_img = Image.new('RGB', (512, 512), color=(255, 100, 100))
        # Add error text to image
        draw = ImageDraw.Draw(error_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((50, 200), f"Error: {str(e)[:50]}...", fill=(0, 0, 0), font=font)
        return error_img

# Create comparison view function
def create_comparison(original, generated, show):
    """Create a side-by-side comparison of original and generated images"""
    if not show or original is None or generated is None:
        return gr.update(visible=False), None

    if isinstance(original, np.ndarray):
        original = Image.fromarray(original)

    # Create a side-by-side comparison
    comparison = Image.new('RGB', (original.width + generated.width, max(original.height, generated.height)))
    comparison.paste(original, (0, 0))
    comparison.paste(generated, (original.width, 0))

    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), "Original", fill=(255, 255, 255), font=font)
    draw.text((original.width + 10, 10), "Generated", fill=(255, 255, 255), font=font)

    return gr.update(visible=True), comparison

# Handle feedback submission
def submit_user_feedback(rating, text):
    """Process user feedback"""
    logger.info(f"Received feedback - Rating: {rating}, Text: {text}")
    return f"Thank you for your feedback! You rated us {rating}/5."

# Fallback function to generate a placeholder image
def generate_placeholder_image():
    """Create a placeholder image with instructions"""
    img = Image.new('RGB', (512, 512), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((120, 200), "Image Generation Placeholder", fill=(100, 100, 100), font=font)
    draw.text((80, 240), "Please check console for error details", fill=(100, 100, 100), font=small_font)
    return img

# Define style modifiers for Similar Image tab
style_modifiers = ["None", "Vintage", "Modern", "Minimalist", "Bohemian", "Elegant", "Sporty", "Punk", "Gothic", "Retro"]

# HTML for welcome screen with improved design
welcome_html = """
<!DOCTYPE html>
<html>
<head>
<style>
  body, html {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    font-family: 'Arial', sans-serif;
  }

  .scene {
    width: 100%;
    height: 100vh;
    background: linear-gradient(135deg, #000428, #004e92);
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .card-container {
    width: 60%;
    height: 60%;
    background: linear-gradient(135deg, #FF8E00, #FF4500);
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    text-align: center;
    flex-direction: column;
    padding: 20px;
  }

  .title {
    font-size: 3em;
    font-weight: bold;
    color: white;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    margin-bottom: 10px;
  }

  .subtitle {
    font-size: 1.5em;
    color: white;
    margin-bottom: 30px;
    max-width: 80%;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
  }

  .features {
    color: white;
    text-align: left;
    margin-bottom: 30px;
    font-size: 1.1em;
  }

  .start-button {
    padding: 15px 40px;
    background: linear-gradient(135deg, #FFF, #E0E0E0);
    color: #FF4500;
    font-size: 1.9em;
    font-weight: bold;
    border: none;
    border-radius: 50px;
    cursor: pointer;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease-in-out;
  }

  .start-button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
  }
</style>
</head>
<body>
  <div class="scene">
    <div class="card-container">
      <h1 class="title">Fashion AI Generator</h1>
      <p class="subtitle">Create stunning fashion designs with the power of artificial intelligence</p>
      <div class="features">
        <p>• Generate custom clothing designs with easy-to-use controls</p>
        <p>• Create similar designs from reference images</p>
        <p>• Try on virtual clothing with our advanced try-on system</p>
      </div>
      <button class="start-button" onclick="document.getElementById('startButtonGradio').click();">START NOW</button>
    </div>
  </div>
</body>
</html>
"""

# Build Gradio UI with Tabs
def build_gradio_interface():
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        # Start Page
        with gr.Group(visible=True) as welcome_page:
            gr.HTML(welcome_html)
            start_btn = gr.Button("Start", visible=False, elem_id="startButtonGradio")

        # Main Application with Tabs (initially hidden)
        with gr.Group(visible=False) as main_app:
            gr.Markdown("# Fashion AI Application")

            with gr.Tabs() as tabs:
                # Tab 1: Fashion Generator
                with gr.TabItem("Fashion Generator"):
                    gr.Markdown("## Generate Custom Fashion Designs")

                    input_type = gr.Radio(choices=["Dropdowns", "Manual Text"], label="Select Input Type", value="Dropdowns")

                    with gr.Row():
                        with gr.Column(scale=1):
                            # Dropdown inputs (visible initially)
                            category = gr.Dropdown(choices=categories, label="Category", value="Dress", visible=True)
                            style = gr.Dropdown(choices=styles, label="Style", value="Modern", visible=True)
                            color_scheme = gr.Dropdown(choices=color_schemes, label="Color Scheme", value="Vibrant", visible=True)
                            pattern = gr.Dropdown(choices=patterns, label="Pattern", value="Plain", visible=True)
                            material = gr.Dropdown(choices=materials, label="Material", value="Cotton", visible=True)
                            fit = gr.Dropdown(choices=fits, label="Fit", value="Regular", visible=True)
                            occasion = gr.Dropdown(choices=occasions, label="Occasion", value="Casual", visible=True)
                            season = gr.Dropdown(choices=seasons, label="Season", value="Summer", visible=True)

                            # Custom text input (hidden initially)
                            custom_prompt = gr.Textbox(
                                label="Custom Description",
                                placeholder="Describe your fashion design...",
                                interactive=True,
                                visible=False
                            )
                            generate_btn = gr.Button("Generate Fashion Image", variant="primary")
                            status_text = gr.Markdown("Ready to generate...")

                        with gr.Column(scale=1):
                            output_image = gr.Image(label="Generated Fashion Design", type="pil")

                    # Feedback Section
                    with gr.Group():
                        gr.Markdown("### We'd Love Your Feedback!")
                        rating = gr.Slider(minimum=1, maximum=5, step=1, label="Rate (1-5)", value=5)
                        feedback_text = gr.Textbox(
                            label="Your Feedback",
                            lines=3,
                            placeholder="Your thoughts on the generated design"
                        )
                        submit_feedback = gr.Button("Submit Feedback")
                        feedback_message = gr.Markdown(visible=True)

                # Tab 2: Similar Image Generator
                with gr.TabItem("Similar Image Generator"):
                    gr.Markdown("## Generate Similar Fashion Images")

                    with gr.Row():
                        with gr.Column(scale=1):
                            reference_image = gr.Image(label="Upload Reference Image", type="numpy")

                            with gr.Group():
                                gr.Markdown("### Modification Options")

                                # Color modification
                                target_color = gr.Dropdown(
                                    choices=["Original"] + colors,
                                    label="Target Color",
                                    value="Original",
                                    info="Change the color of the generated item"
                                )

                                # Style modification
                                style_modifier = gr.Dropdown(
                                    choices=style_modifiers,
                                    label="Style Modifier",
                                    value="None",
                                    info="Apply a style to the generated image"
                                )

                                # Similarity strength with improved description
                                similarity_strength = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    step=0.1,
                                    label="Transformation Strength",
                                    value=0.7,
                                    info="Lower = more similar to original, Higher = more creative"
                                )

                                # Additional options as checkboxes
                                with gr.Row():
                                    maintain_structure = gr.Checkbox(
                                        label="Maintain Structure",
                                        value=True,
                                        info="Keep the overall structure of the original item"
                                    )
                                    enhance_details = gr.Checkbox(
                                        label="Enhance Details",
                                        value=False,
                                        info="Add more intricate details to the generated image"
                                    )

                            similar_generate_btn = gr.Button("Generate Similar Image", variant="primary")
                            similar_status_text = gr.Markdown("Upload an image and adjust options...")

                        with gr.Column(scale=1):
                            similar_output = gr.Image(label="Generated Similar Image", type="pil")

                            # Comparison view toggle
                            show_comparison = gr.Checkbox(label="Show Before/After Comparison", value=False)

                            # Container for comparison view (initially empty)
                            comparison_output = gr.Image(label="Comparison View", visible=False)

                # Tab 3: Virtual Try-On (Placeholder for future development)
                with gr.TabItem("Virtual Try-On"):
                    gr.Markdown("## Virtual Try-On System")
                    gr.Markdown("This feature is coming soon! Check back later for updates.")

        # Event Handlers
        # Welcome page to main app transition
        start_btn.click(
            lambda: [gr.update(visible=False), gr.update(visible=True)],
            outputs=[welcome_page, main_app]
        )

        # Toggle between dropdown inputs and custom prompt based on radio button selection
        input_type.change(
            toggle_input_type,
            inputs=[input_type],
            outputs=[category, style, color_scheme, pattern, material, fit, occasion, season, custom_prompt]
        )

        # Tab 1: Fashion Generator
        # Two-step process for better UI feedback
        def safe_get_completion(*args):
            try:
                return "Generating image, please wait...", None, gr.update(interactive=False)
            except Exception as e:
                logger.error(f"Error starting generation: {str(e)}")
                return f"Error: {str(e)}", generate_placeholder_image(), gr.update(interactive=True)

        def complete_generation(*args):
            try:
                result = get_completion(*args)
                return "Image generated successfully!", result, gr.update(interactive=True)
            except Exception as e:
                logger.error(f"Error in image generation: {str(e)}")
                return f"Error: {str(e)}", generate_placeholder_image(), gr.update(interactive=True)

        generate_btn.click(
            safe_get_completion,
            inputs=[],
            outputs=[status_text, output_image, generate_btn]
        ).then(
            complete_generation,
            inputs=[
                input_type, category, style, color_scheme, pattern,
                material, fit, occasion, season, custom_prompt
            ],
            outputs=[status_text, output_image, generate_btn]
        )

        submit_feedback.click(
            submit_user_feedback,
            inputs=[rating, feedback_text],
            outputs=feedback_message
        )

        # Tab 2: Similar Image Generator
        # Two-step process for better UI feedback
        def start_similar_generation():
            return "Generating similar image, please wait...", None, gr.update(interactive=False)

        def complete_similar_generation(reference_image, strength, target_color, style_modifier, maintain_structure, enhance_details):
            try:
                result = generate_similar_image(
                    reference_image, 
                    strength, 
                    target_color, 
                    style_modifier, 
                    maintain_structure, 
                    enhance_details
                )
                return "Similar image generated successfully!", result, gr.update(interactive=True)
            except Exception as e:
                logger.error(f"Error in similar image generation: {str(e)}")
                return f"Error: {str(e)}", generate_placeholder_image(), gr.update(interactive=True)

        similar_generate_btn.click(
            start_similar_generation,
            outputs=[similar_status_text, similar_output, similar_generate_btn]
        ).then(
            complete_similar_generation,
            inputs=[reference_image, similarity_strength, target_color, style_modifier, maintain_structure, enhance_details],
            outputs=[similar_status_text, similar_output, similar_generate_btn]
        )

        # Handle comparison view toggle
        show_comparison.change(
            create_comparison,
            inputs=[reference_image, similar_output, show_comparison],
            outputs=[comparison_output, comparison_output]
        )

        return demo

# Load models (can be done outside the Gradio interface for improved startup time)
# txt2img_pipeline, img2img_pipeline = load_models()  # Uncomment when ready to use

def main():
    try:
        # Initialize models
        global txt2img_pipeline, img2img_pipeline
        txt2img_pipeline, img2img_pipeline = load_models()
        
        # Build and launch the Gradio interface
        logger.info("Starting Fashion AI Generator application")
        demo = build_gradio_interface()
        demo.launch(share=False)  # Set share=True if you want a public link
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()