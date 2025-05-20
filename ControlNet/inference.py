import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import config  # your config module with save_memory flag, etc.

# ————————————————————————————————————————————————
# 1) Initialize everything once
# ————————————————————————————————————————————————
apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
state_dict = load_state_dict(
    '/home/sauravdosi/mediffuse/ControlNet/lightning_logs/version_0/checkpoints/epoch=61-step=37138.ckpt',
    location='cuda'
)
model.load_state_dict(state_dict)
model = model.cuda()

ddim_sampler = DDIMSampler(model)


# ————————————————————————————————————————————————
# 2) Inference function
# ————————————————————————————————————————————————
def process(
    input_image: np.ndarray,
    prompt: str,
    a_prompt: str = 'translate CT scan image to MRI scan image',
    n_prompt: str = ('CT scan like'),
    num_samples: int = 1,
    image_resolution: int = 512,
    ddim_steps: int = 20,
    guess_mode: bool = False,
    strength: float = 1.0,
    scale: float = 9.0,
    seed: int = -1,
    eta: float = 0.0,
    low_threshold: int = 100,
    high_threshold: int = 200
):
    """
    Returns a list whose first element is the inverted Canny map
    and the rest are generated samples.
    """
    with torch.no_grad():
        # Resize and Canny
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, _ = img.shape
        detected = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected)

        # Prepare control tensor
        control = torch.from_numpy(detected_map).float().cuda() / 255.0
        control = control.unsqueeze(0).repeat(num_samples, 1, 1, 1)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        # Seed RNG
        if seed == -1:
            seed = random.randint(0, 2**16 - 1)
        seed_everything(seed)

        # VRAM optimization
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Conditioning
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

        # Control scales
        if guess_mode:
            model.control_scales = [
                strength * (0.825 ** float(12 - i)) for i in range(13)
            ]
        else:
            model.control_scales = [strength] * 13

        # Sampling
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

        # Decode
        x_samples = model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c')
            * 127.5 + 127.5
        ).cpu().numpy().clip(0, 255).astype(np.uint8)

        # Return inverted Canny + samples
        return [255 - detected_map] + [x_samples[i] for i in range(num_samples)]


# ————————————————————————————————————————————————
# 3) Example usage (edit paths & params here)
# ————————————————————————————————————————————————
if __name__ == '__main__':
    # --- User settings ---
    input_path      = '/home/sauravdosi/mediffuse/presentation/0042_CT_contours.png'
    prompt          = 'translate CT scan image to MRI scan image'
    num_samples     = 2
    image_resolution= 512
    ddim_steps      = 50
    guess_mode      = False
    strength        = 0.85
    scale           = 9.0
    seed            = 42
    eta             = 0.0
    low_threshold   = 100
    high_threshold  = 200

    # 1) Load image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {input_path}")

    # 2) Run inference
    outputs = process(
        img, prompt,
        num_samples=num_samples,
        image_resolution=image_resolution,
        ddim_steps=ddim_steps,
        guess_mode=guess_mode,
        strength=strength,
        scale=scale,
        seed=seed,
        eta=eta,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )

    # 3) Save results
    base = input_path.rsplit('.', 1)[0]
    cv2.imwrite(f"{base}_canny_inv.png", outputs[0])
    for idx, sample in enumerate(outputs[1:], start=1):
        print("hey")
        cv2.imwrite(f"{base}_sample_{idx}.png", sample)

    print(f"Saved {len(outputs)} images to disk.")
