import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline
import gradio as gr
import numpy as np

# Configuration
MODEL_PATH = "/home/sauravdosi/mediffuse/models/instruct-ct2mri-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load pipeline
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# Inference with diffusion visualization
def translate_ct_to_mri(
    input_image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    image_guidance_scale: float,
    guidance_scale: float,
    seed: int
) -> list:
    img = ImageOps.exif_transpose(input_image).convert("RGB")
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    intermediates = []

    def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
        # Extract latents tensor
        latents = callback_kwargs.get('latents', callback_kwargs.get('sample'))
        with torch.no_grad():
            decoded = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor)
            # decoded.sample shape: [batch, C, H, W]
            image_tensor = decoded.sample[0]
        # Convert tensor to NumPy array (CHW -> HWC)
        arr = image_tensor.cpu().permute(1, 2, 0).numpy()
        # Scale to [0,255] uint8
        arr = ((arr + 1) / 2 * 255.0).round().clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        intermediates.append(pil_img)
        # Return kwargs to avoid NoneType
        return callback_kwargs

    # Run pipeline
    output = pipe(
        prompt=prompt,
        image=img,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator,
        callback_on_step_end=callback_on_step_end
    ).images[0]

    intermediates.append(output)
    return intermediates

# Gradio interface
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# CT to MRI Translation with Diffusion Visualization")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input CT Image")
                prompt = gr.Textbox(
                    value="translate CT scan image to MRI scan image",
                    label="Prompt"
                )
                num_steps = gr.Slider(1, 100, step=1, value=50, label="Inference Steps")
                img_guidance = gr.Slider(0.0, 5.0, step=0.1, value=1.5, label="Image Guidance Scale")
                txt_guidance = gr.Slider(0.0, 20.0, step=0.5, value=7.5, label="Text Guidance Scale")
                seed = gr.Number(value=42, precision=0, label="Seed")
                run_btn = gr.Button("Run")
            with gr.Column():
                gallery = gr.Gallery(label="Diffusion Steps", columns=5, height="auto")

        run_btn.click(
            translate_ct_to_mri,
            inputs=[input_image, prompt, num_steps, img_guidance, txt_guidance, seed],
            outputs=gallery
        )

    demo.launch(share=True)

if __name__ == "__main__":
    main()
