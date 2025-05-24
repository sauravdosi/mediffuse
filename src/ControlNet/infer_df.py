#!/usr/bin/env python3
import os
import time
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
# —————————————————————————————————————————————————————————
# 1) IMPORT YOUR INFERENCE FUNCTION & SETUP
# —————————————————————————————————————————————————————————
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import einops
import random
from pytorch_lightning import seed_everything
import config

# initialize once
apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
state_dict = load_state_dict(
    '/home/sauravdosi/mediffuse/ControlNet/lightning_logs/version_0/checkpoints/epoch=61-step=37138.ckpt',
    location='cuda'
)
model.load_state_dict(state_dict)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def process(
    input_image: np.ndarray,
    prompt: str = "translate CT scan image to MRI scan image",
        a_prompt: str = 'translate CT scan image to MRI scan image',
        n_prompt: str = ('CT scan like'),
    image_resolution: int = 512,
    ddim_steps: int = 50,
    guess_mode: bool = False,
    strength: float = 0.85,
    scale: float = 7.5,
    seed: int = 42,
    eta: float = 0.0,
    low_threshold: int = 100,
    high_threshold: int = 200,
num_samples: int = 1,
):
    """
    Returns the first generated sample (RGB uint8) and the elapsed inference time.
    """
    # prepare
    img = resize_image(HWC3(input_image), image_resolution)
    H, W, _ = img.shape
    detected = apply_canny(img, low_threshold, high_threshold)
    detected_map = HWC3(detected)

    control = torch.from_numpy(detected_map).float().cuda() / 255.0
    control = control.unsqueeze(0).repeat(num_samples, 1, 1, 1)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    # timing start
    torch.cuda.synchronize()
    t0 = time.time()

    # seed
    seed_everything(seed)

    # low-vram
    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    cond = {
        "c_concat": [control],
        "c_crossattn": [
            model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)
        ]
    }
    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn": [
            model.get_learned_conditioning([n_prompt] * num_samples)
        ]
    }
    shape = (4, H // 8, W // 8)
    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    if guess_mode:
        model.control_scales = [
            strength * (0.825 ** float(12 - i)) for i in range(13)
        ]
    else:
        model.control_scales = [strength] * 13

    samples, _ = ddim_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond
    )
    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    torch.cuda.synchronize()
    t1 = time.time()

    x_samples = model.decode_first_stage(samples)
    x_samples = (
        einops.rearrange(x_samples, 'b c h w -> b h w c')
        * 127.5 + 127.5
    ).cpu().numpy().clip(0, 255).astype(np.uint8)

    # return first sample + elapsed
    return x_samples[0], (t1 - t0)


# —————————————————————————————————————————————————————————
# 2) EVAL LOOP
# —————————————————————————————————————————————————————————
if __name__ == '__main__':
    LOCAL_CT_DIR = "/home/sauravdosi/mediffuse/data/test_ct2mri_img2img/CT"
    LOCAL_GT_DIR = "/home/sauravdosi/mediffuse/data/test_ct2mri_img2img/MRI"
    OUTPUT_DIR = "test_results_ct2mri_time_controlnet"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    ct_files = sorted(f for f in os.listdir(LOCAL_CT_DIR)
                      if f.lower().endswith(('.png','.jpg','.jpeg')))

    for idx, fname in enumerate(tqdm(ct_files, desc="Evaluating")):
        stem = os.path.splitext(fname)[0]
        ct_path = os.path.join(LOCAL_CT_DIR, fname)
        gt_path = os.path.join(LOCAL_GT_DIR, fname)
        if not os.path.exists(gt_path):
            gt_path = os.path.join(
                LOCAL_GT_DIR, stem.replace('_ct','') + '_mr.png'
            )
        if not os.path.exists(gt_path):
            print(f"  ⚠️  Skipping {fname}: no GT found")
            continue

        # load & preprocess
        ct_img = cv2.imread(ct_path)
        ct_img = cv2.cvtColor(ct_img, cv2.COLOR_BGR2RGB)
        ct_img = cv2.resize(ct_img, (512,512))
        gt_img = Image.open(gt_path).convert("RGB").resize((512,512))

        # inference
        pred_arr, inf_time = process(ct_img)
        pred_img = Image.fromarray(pred_arr)

        # save
        ct_save = os.path.join(OUTPUT_DIR, f"{idx:05d}_ct.png")
        gt_save = os.path.join(OUTPUT_DIR, f"{idx:05d}_gt.png")
        pred_save = os.path.join(OUTPUT_DIR, f"{idx:05d}_pred.png")
        cv2.imwrite(ct_save, cv2.cvtColor(ct_img, cv2.COLOR_RGB2BGR))
        gt_img.save(gt_save)
        pred_img.save(pred_save)

        # metrics
        gt_arr = np.array(gt_img)
        psnr_val = peak_signal_noise_ratio(
            gt_arr, pred_arr,
            data_range=gt_arr.max()-gt_arr.min()
        )
        ssim_val = structural_similarity(
            gt_arr, pred_arr,
            data_range=gt_arr.max()-gt_arr.min(),
            channel_axis=2
        )

        results.append({
            "id": idx,
            "filename": fname,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "inference_time": inf_time
        })

    # save CSV + summary row
    df = pd.DataFrame(results)
    mean_vals = df[["psnr","ssim","inference_time"]].mean().rename({
        "psnr": "psnr_mean",
        "ssim": "ssim_mean",
        "inference_time": "inference_time_mean"
    })
    df = pd.concat([df, mean_vals.to_frame().T], ignore_index=True)
    csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Done! Results and images under `{OUTPUT_DIR}/`, metrics in `{csv_path}`.")
