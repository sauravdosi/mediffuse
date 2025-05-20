import os
import shutil
import json
import argparse
import random

def prepare_dataset(input_dir, output_dir, prompt="translate CT scan contour image to MRI scan image", max_pairs=None):
    ct_dir = os.path.join(output_dir, "CT")
    mr_dir = os.path.join(output_dir, "MRI")
    os.makedirs(ct_dir, exist_ok=True)
    os.makedirs(mr_dir, exist_ok=True)

    # Step 1: Collect all valid image numbers
    files = os.listdir(input_dir)
    nums = set()
    for fname in files:
        if fname.lower().endswith('_ctcontour.png'):
            nums.add(fname[:-14])  # Remove '_ct.png'
        elif fname.lower().endswith('_mr.png'):
            nums.add(fname[:-7])  # Remove '_mr.png'

    all_pairs = sorted(list(nums))
    if max_pairs is not None:
        all_pairs = random.sample(all_pairs, min(max_pairs, len(all_pairs)))

    metadata = []
    for num in all_pairs:
        ct_name = f"{num}_ctcontour.png"
        mr_name = f"{num}_mr.png"
        ct_src = os.path.join(input_dir, ct_name)
        mr_src = os.path.join(input_dir, mr_name)

        if not os.path.exists(ct_src) or not os.path.exists(mr_src):
            continue

        shutil.copy2(ct_src, os.path.join(ct_dir, ct_name))
        shutil.copy2(mr_src, os.path.join(mr_dir, mr_name))

        metadata.append({
            "input_image": os.path.join("CT", ct_name),
            "edited_image": os.path.join("MRI", mr_name),
            "edit_prompt": prompt
        })

    with open(os.path.join(output_dir, "metadata.jsonl"), 'w') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"Prepared {len(metadata)} pairs to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="translate CT scan contour image to MRI scan image")
    parser.add_argument("--max_pairs", type=int, default=None, help="Maximum number of pairs to use")
    args = parser.parse_args()
    prepare_dataset(args.input_dir, args.output_dir, args.prompt, args.max_pairs)
