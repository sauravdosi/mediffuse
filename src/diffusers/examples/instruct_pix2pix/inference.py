import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline
import argparse

parser = argparse.ArgumentParser()
model_path = "./instruct-mri2ct-large"
input_image = "/home/sauravdosi/mediffuse/presentation/quiz/00087_mr.png"
prompt = "translate MRI scan image to CT scan image"
output = "/home/sauravdosi/mediffuse/presentation/ip2p_mr2ct.png"
num_inference_steps = 50
image_guidance_scale = 1.5
guidance_scale = 7.5
seed = 42

# 1. Load pipeline
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_path, torch_dtype=torch.float16
).to("cuda")

# 2. Load & preprocess input
img = Image.open(input_image)
img = ImageOps.exif_transpose(img).convert("RGB")

# 3. Run inference
generator = torch.Generator("cuda").manual_seed(seed)
edited = pipe(
    prompt=prompt,
    image=img,
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
    guidance_scale=guidance_scale,
    generator=generator,
).images[0]

# 4. Save result
edited.save(output)
print(f"✅ Saved: {output}")
