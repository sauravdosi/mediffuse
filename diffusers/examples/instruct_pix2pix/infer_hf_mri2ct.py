#!/usr/bin/env python3
import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from diffusers import StableDiffusionInstructPix2PixPipeline
from datasets import load_dataset

# === Configuration ===
MODEL_PATH = "./instruct-mri2ct-large"
DATASET_NAME = "sauravdosi/mri2ct-img2img-train"
SPLIT = "train"
OUTPUT_DIR = "train_results_mri2ct"
RESOLUTION = (512, 512)
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_GUIDANCE_SCALE = 1.5
SEED = 42

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load fine-tuned model
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16
).to("cuda")

# Load dataset split
dataset = load_dataset(DATASET_NAME, split=SPLIT)

# Prepare result storage
results = []

# Evaluation loop
for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
    # Load and preprocess CT image
    ct_item = example.get("input_image")
    if isinstance(ct_item, dict) and "path" in ct_item:
        ct_img = Image.open(ct_item["path"])
    else:
        ct_img = ct_item
    ct_img = ImageOps.exif_transpose(ct_img).convert("RGB").resize(RESOLUTION)

    # Load and preprocess ground-truth MRI
    gt_item = example.get("edited_image")
    if isinstance(gt_item, dict) and "path" in gt_item:
        gt_img = Image.open(gt_item["path"])
    else:
        gt_img = gt_item
    gt_img = ImageOps.exif_transpose(gt_img).convert("RGB").resize(RESOLUTION)

    # Save input CT and GT MRI for visual verification
    ct_save_path = os.path.join(OUTPUT_DIR, f"{idx:05d}_gt.png")
    gt_save_path = os.path.join(OUTPUT_DIR, f"{idx:05d}_mr.png")
    ct_img.save(ct_save_path)
    gt_img.save(gt_save_path)

    # Get prompt
    prompt = example.get("edit_prompt", "translate CT to MRI")

    # Inference
    generator = torch.Generator("cuda").manual_seed(SEED + idx)
    pred_img = pipe(
        prompt=prompt,
        image=ct_img,
        num_inference_steps=NUM_INFERENCE_STEPS,
        image_guidance_scale=IMAGE_GUIDANCE_SCALE,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0].resize(RESOLUTION)

    # Save generated image
    pred_save_path = os.path.join(OUTPUT_DIR, f"{idx:05d}_pred.png")
    pred_img.save(pred_save_path)

    # Compute metrics
    gt_arr = np.array(gt_img)
    pred_arr = np.array(pred_img)
    psnr_val = peak_signal_noise_ratio(
        gt_arr, pred_arr,
        data_range=gt_arr.max() - gt_arr.min()
    )
    ssim_val = structural_similarity(
        gt_arr, pred_arr,
        data_range=gt_arr.max() - gt_arr.min(),
        channel_axis=-1
    )

    results.append({"image_id": idx, "psnr": psnr_val, "ssim": ssim_val})

# Aggregate to DataFrame
df = pd.DataFrame(results)
mean_vals = df.mean(numeric_only=True).rename({"psnr": "psnr_mean", "ssim": "ssim_mean"})
df = pd.concat([df, mean_vals.to_frame().T], ignore_index=True)

# Save metrics to CSV
metrics_csv = os.path.join(OUTPUT_DIR, "metrics.csv")
df.to_csv(metrics_csv, index=False)
print(f"✅ Metrics saved to {metrics_csv}")
print(f"✅ Saved CT, GT, and generated images for {len(results)} samples in '{OUTPUT_DIR}/' ")
