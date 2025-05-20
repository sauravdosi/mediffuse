#!/usr/bin/env python3
import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from diffusers import StableDiffusionInstructPix2PixPipeline
from datasets import load_dataset

# === Configuration ===
MODEL_PATH = "./instruct-ct2mri-large"
LOCAL_CT_DIR = "/home/sauravdosi/mediffuse/data/test_ct2mri_img2img/CT"
LOCAL_GT_DIR = "/home/sauravdosi/mediffuse/data/test_ct2mri_img2img/MRI"
DATASET_NAME = ""
SPLIT = "test"

OUTPUT_DIR = "test_results_ct2mri_time"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load pipeline once
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16
).to("cuda")

results = []

def run_inference(ct_img, prompt):
    # ensure all CUDA ops are done before starting timing
    torch.cuda.synchronize()
    start = time.time()
    output = pipe(
        prompt=prompt,
        image=ct_img,
        num_inference_steps=50,
        image_guidance_scale=1.5,
        guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(42)
    )
    # wait for all kernels to finish
    torch.cuda.synchronize()
    end = time.time()
    # return PIL image and elapsed time
    return output.images[0], end - start

if LOCAL_CT_DIR:
    ct_files = sorted([f for f in os.listdir(LOCAL_CT_DIR)
                       if f.lower().endswith(('.png','.jpg','.jpeg'))])
    for idx, fname in enumerate(tqdm(ct_files, desc="Local eval")):
        stem = os.path.splitext(fname)[0]
        ct_path = os.path.join(LOCAL_CT_DIR, fname)
        gt_candidate = os.path.join(LOCAL_GT_DIR, fname)
        if not os.path.exists(gt_candidate):
            gt_candidate = os.path.join(
                LOCAL_GT_DIR, stem.replace('_ct','') + '_mr.png'
            )
        if not os.path.exists(gt_candidate):
            print(f"Skipping {fname}: no GT found")
            continue

        ct_img = Image.open(ct_path)
        ct_img = ImageOps.exif_transpose(ct_img).convert("RGB").resize((512,512))
        gt_img = Image.open(gt_candidate).convert("RGB").resize((512,512))

        prompt = "translate CT scan image to MRI scan image"
        pred_img, inf_time = run_inference(ct_img, prompt)
        pred_img = pred_img.resize((512,512))

        # Save images
        ct_img.save(os.path.join(OUTPUT_DIR, f"{idx:05d}_ct.png"))
        gt_img.save(os.path.join(OUTPUT_DIR, f"{idx:05d}_gt.png"))
        pred_img.save(os.path.join(OUTPUT_DIR, f"{idx:05d}_pred.png"))

        # Compute metrics
        gt_arr = np.array(gt_img)
        pred_arr = np.array(pred_img)
        psnr_val = peak_signal_noise_ratio(gt_arr, pred_arr,
                                           data_range=gt_arr.max()-gt_arr.min())
        ssim_val = structural_similarity(gt_arr, pred_arr,
                                         data_range=gt_arr.max()-gt_arr.min(),
                                         channel_axis=2)

        results.append({
            "id": idx,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "inference_time": inf_time
        })

else:
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    for idx, ex in enumerate(tqdm(ds, desc="HF eval")):
        ct_img = ex["input_image"].resize((512,512))
        gt_img = ex["edited_image"].resize((512,512))
        prompt = ex["edit_prompt"]

        pred_img, inf_time = run_inference(ct_img, prompt)
        pred_img = pred_img.resize((512,512))

        # Save images
        ct_img.save(os.path.join(OUTPUT_DIR, f"{idx:05d}_ct.png"))
        gt_img.save(os.path.join(OUTPUT_DIR, f"{idx:05d}_gt.png"))
        pred_img.save(os.path.join(OUTPUT_DIR, f"{idx:05d}_pred.png"))

        # Compute metrics
        gt_arr = np.array(gt_img)
        pred_arr = np.array(pred_img)
        psnr_val = peak_signal_noise_ratio(gt_arr, pred_arr,
                                           data_range=gt_arr.max()-gt_arr.min())
        ssim_val = structural_similarity(gt_arr, pred_arr,
                                         data_range=gt_arr.max()-gt_arr.min(),
                                         channel_axis=2)

        results.append({
            "id": idx,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "inference_time": inf_time
        })

# Summarize metrics
df = pd.DataFrame(results)
mean_vals = df.mean().rename({
    "psnr": "psnr_mean",
    "ssim": "ssim_mean",
    "inference_time": "inference_time_mean"
})
df = pd.concat([df, mean_vals.to_frame().T], ignore_index=True)
df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

import ace_tools as tools
tools.display_dataframe_to_user(
    name="CT→MRI PSNR, SSIM & Inference Time", dataframe=df
)
print(f"✅ Results + images saved under `{OUTPUT_DIR}/`")
