#!/usr/bin/env python3
"""
Prepare and push a Hugging Face-style dataset for CT→MRI training in ControlNet format,
mirroring raulc0399/open_pose_controlnet layout, then upload it as a private Hugging Face dataset.

Set the directory paths below before running.
"""
import os
import json
import shutil
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Image
from huggingface_hub import create_repo

# === Configuration ===
CT_DIR = "/home/sauravdosi/mediffuse/data/train_img2img/CT"     # CT images folder
MRI_DIR = "/home/sauravdosi/mediffuse/data/train_img2img/MRI"   # MRI images folder
OUT_DIR = "/home/sauravdosi/mediffuse/data/train_flux"          # root for dataset files
DATASET_REPO = "sauravdosi/ct2mri-flux"                         # HF dataset repo
PROMPT = "translate CT scan image to a more detailed MRI scan image"
PRIVATE = False

# Supported extensions
ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

def is_image(path: Path) -> bool:
    return path.suffix.lower() in ext

# Create train/images and train/conditioning_images
root = Path(OUT_DIR)
train_img = root / 'train' / 'images'
train_cond = root / 'train' / 'conditioning_images'
train_img.mkdir(parents=True, exist_ok=True)
train_cond.mkdir(parents=True, exist_ok=True)

# Collect CT and MRI files by key
from_ct = {p.stem[:-3]: p for p in Path(CT_DIR).iterdir() if is_image(p) and p.stem.endswith('_ct')}
from_mri = {p.stem[:-3]: p for p in Path(MRI_DIR).iterdir() if is_image(p) and p.stem.endswith('_mr')}

# Build metadata and copy files
metadata = []
cnt = 0
for key in sorted(from_ct):
    ct_path = from_ct[key]
    mri_path = from_mri.get(key)
    if mri_path is None:
        print(f"Warning: No MRI for CT {ct_path.name}")
        continue
    # Copy
    dst_ct = train_img / ct_path.name
    dst_mri = train_cond / mri_path.name
    shutil.copy2(ct_path, dst_ct)
    shutil.copy2(mri_path, dst_mri)
    metadata.append({
        "image": f"train/images/{dst_ct.name}",
        "conditioning_image": f"train/conditioning_images/{dst_mri.name}",
        "text": PROMPT
    })
    cnt += 1
print(f"Prepared {cnt} pairs under {root / 'train'}")

# Write metadata.jsonl
meta_file = root / 'metadata.jsonl'
with open(meta_file, 'w') as f:
    for entry in metadata:
        f.write(json.dumps(entry) + '\n')
print(f"Wrote metadata to {meta_file}")

# Create HF Dataset locally
meta_df = pd.read_json(meta_file, lines=True)
features = Features({
    "image": Image(),
    "conditioning_image": Image(),
    "text": Value("string"),
})
dataset = Dataset.from_pandas(meta_df, features=features)
ds_dict = DatasetDict({"train": dataset})

# Save Parquet for backup
data_dir = root / 'data'
data_dir.mkdir(exist_ok=True)
parquet_file = data_dir / 'ct2mri.parquet'
ds_dict['train'].to_parquet(parquet_file)
print(f"Saved Parquet snapshot to {parquet_file}")

# Change working dir so relative paths resolve
os.chdir(OUT_DIR)

# Push to HF Hub
create_repo(DATASET_REPO, repo_type="dataset", private=PRIVATE)
ds_dict.push_to_hub(DATASET_REPO, private=PRIVATE)
print(f"Pushed dataset to https://huggingface.co/datasets/{DATASET_REPO}")
