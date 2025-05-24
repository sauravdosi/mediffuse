import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline
import gradio as gr
import numpy as np
import queue
import threading
from typing import Iterator

# Configuration
MODEL_PATH = "/home/sauravdosi/mediffuse/models/instruct-mri2ct-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load pipeline
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# Modified inference function with gradual yield
def translate_ct_to_mri(
    input_image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    image_guidance_scale: float,
    guidance_scale: float,
    seed: int
) -> Iterator[list[Image.Image]]:
    img = ImageOps.exif_transpose(input_image).convert("RGB")
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    q = queue.Queue()

    def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
        latents = callback_kwargs.get('latents', callback_kwargs.get('sample'))
        with torch.no_grad():
            decoded = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor)
            image_tensor = decoded.sample[0]
        arr = image_tensor.cpu().permute(1, 2, 0).numpy()
        arr = ((arr + 1) / 2 * 255.0).round().clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        q.put(pil_img)

    def run_pipeline():
        try:
            output = pipe(
                prompt=prompt,
                image=img,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=callback_on_step_end
            ).images[0]
            q.put(output)
        finally:
            q.put(None)  # Completion signal

    thread = threading.Thread(target=run_pipeline)
    thread.start()

    intermediates = []
    while True:
        item = q.get()
        if item is None:
            break
        intermediates.append(item)
        yield intermediates

    thread.join()

# Gradio interface with streaming support
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
                gallery = gr.Gallery(label="Diffusion Process", columns=5, height="auto", object_fit="contain")

        run_btn.click(
            translate_ct_to_mri,
            inputs=[input_image, prompt, num_steps, img_guidance, txt_guidance, seed],
            outputs=gallery
        )

    demo.launch(share=True)

if __name__ == "__main__":
    main()