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
# To evaluate a local directory of CT/GT images, set these:
LOCAL_MR_DIR = "/home/sauravdosi/mediffuse/data/test_ct2mri_img2img/MRI"    # directory containing CT images
LOCAL_GT_DIR = "/home/sauravdosi/mediffuse/data/test_ct2mri_img2img/CT"     # directory containing ground-truth MRI images
# Otherwise, set LOCAL_MR_DIR = "" to use the HF dataset below:
DATASET_NAME = ""
SPLIT = "test"

# Output directory for generated and saved images
OUTPUT_DIR = "test_results_mri2ct"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load pipeline once
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16
).to("cuda")

results = []

if LOCAL_MR_DIR:
    # Local directory mode
    ct_files = sorted([f for f in os.listdir(LOCAL_MR_DIR)
                       if f.lower().endswith(('.png','.jpg','.jpeg'))])
    for idx, fname in enumerate(tqdm(ct_files, desc="Local eval")):
        stem = os.path.splitext(fname)[0]
        ct_path = os.path.join(LOCAL_MR_DIR, fname)
        # determine GT filename (same name or replace suffix)
        gt_candidate = os.path.join(LOCAL_GT_DIR, fname)
        if not os.path.exists(gt_candidate):
            # try replace '_ct' with '_mr'
            gt_candidate = os.path.join(
                LOCAL_GT_DIR, stem.replace('_mr','') + '_ct.png'
            )
        if not os.path.exists(gt_candidate):
            print(f"Skipping {fname}: no GT found")
            continue

        # Load images
        ct_img = Image.open(ct_path)
        ct_img = ImageOps.exif_transpose(ct_img).convert("RGB").resize((512,512))
        gt_img = Image.open(gt_candidate).convert("RGB").resize((512,512))

        # Use default prompt
        prompt = "translate MRI scan image to CT scan image"

        # Inference
        pred = pipe(
            prompt=prompt,
            image=ct_img,
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0].resize((512,512))

        # Save CT, GT, prediction
        ct_img.save(os.path.join(OUTPUT_DIR, f"{idx:05d}_ct.png"))
        gt_img.save(os.path.join(OUTPUT_DIR, f"{idx:05d}_gt.png"))
        pred.save(os.path.join(OUTPUT_DIR, f"{idx:05d}_pred.png"))

        # Metrics
        gt_arr = np.array(gt_img)
        pred_arr = np.array(pred)
        psnr_val = peak_signal_noise_ratio(gt_arr, pred_arr,
                                          data_range=gt_arr.max()-gt_arr.min())
        ssim_val = structural_similarity(gt_arr, pred_arr,
                                         data_range=gt_arr.max()-gt_arr.min(),
                                         channel_axis=2)
        results.append({"id": idx, "psnr": psnr_val, "ssim": ssim_val})

else:
    # HF dataset mode
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    for idx, ex in enumerate(tqdm(ds, desc="HF eval")):
        ct_img = ex["input_image"].resize((512,512))
        gt_img = ex["edited_image"].resize((512,512))
        prompt = ex["edit_prompt"]

        pred = pipe(
            prompt=prompt,
            image=ct_img,
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0].resize((512,512))

        ct_img.save(os.path.join(OUTPUT_DIR, f"{idx:05d}_ct.png"))
        gt_img.save(os.path.join(OUTPUT_DIR, f"{idx:05d}_gt.png"))
        pred.save(os.path.join(OUTPUT_DIR, f"{idx:05d}_pred.png"))

        gt_arr = np.array(gt_img)
        pred_arr = np.array(pred)
        psnr_val = peak_signal_noise_ratio(gt_arr, pred_arr,
                                          data_range=gt_arr.max()-gt_arr.min())
        ssim_val = structural_similarity(gt_arr, pred_arr,
                                         data_range=gt_arr.max()-gt_arr.min(),
                                         channel_axis=2)
        results.append({"id": idx, "psnr": psnr_val, "ssim": ssim_val})

# Summarize metrics
df = pd.DataFrame(results)
mean_vals = df.mean().rename({"psnr": "psnr_mean", "ssim": "ssim_mean"})
df = pd.concat([df, mean_vals.to_frame().T], ignore_index=True)
df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
import ace_tools as tools

tools.display_dataframe_to_user(
    name="CT→MRI PSNR & SSIM Evaluation", dataframe=df)
print(f"✅ Results + images saved under `{OUTPUT_DIR}/`")
