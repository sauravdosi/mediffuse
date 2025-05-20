import os
from PIL import Image

# --- CONFIGURATION ---
input_dir = '/home/sauravdosi/mediffuse/data/raw'  # folder containing your 512×256 images
output_ct = '/home/sauravdosi/mediffuse/data/raw/CT'  # where to save the CT (left) halves
output_mri = '/home/sauravdosi/mediffuse/data/raw/MRI'  # where to save the MRI (right) halves
# ----------------------

# create output folders if they don't exist
os.makedirs(output_ct, exist_ok=True)
os.makedirs(output_mri, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        continue

    path = os.path.join(input_dir, fname)
    img = Image.open(path)
    w, h = img.size

    # sanity check
    if (w, h) != (512, 256):
        print(f"Skipping {fname}: unexpected size {w}×{h}")
        continue

    # crop into left (CT) and right (MRI) halves
    left_box = (0, 0, w // 2, h)  # (x1,y1,x2,y2)
    right_box = (w // 2, 0, w, h)
    ct_img = img.crop(left_box)
    mri_img = img.crop(right_box)

    # resize each to 512×512
    ct_img = ct_img.resize((512, 512), Image.BILINEAR)
    mri_img = mri_img.resize((512, 512), Image.BILINEAR)

    # rotate 90° anticlockwise
    ct_img = ct_img.rotate(90, expand=True)
    mri_img = mri_img.rotate(90, expand=True)

    # build output filenames
    base = os.path.splitext(fname)[0]
    ct_path = os.path.join(output_ct, f"{base}_CT.png")
    mri_path = os.path.join(output_mri, f"{base}_MRI.png")

    # save
    ct_img.save(ct_path)
    mri_img.save(mri_path)

    print(f"Processed {fname} → {os.path.basename(ct_path)}, {os.path.basename(mri_path)}")
