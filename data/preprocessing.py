import os
import nibabel as nib
import numpy as np
from skimage.transform import resize
from PIL import Image
import cv2

def normalize(volume, vmin=None, vmax=None):
    if vmin is not None and vmax is not None:
        volume = np.clip(volume, vmin, vmax)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume

def generate_contour(mask_slice):
    mask_bin = (mask_slice > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(mask_bin) * 255
    if len(contours) == 0:
        return None
    cv2.drawContours(contour_img, contours, -1, 255, 1)
    return contour_img

def overlay_image_with_contour(image, contour):
    image = (image * 255).astype(np.uint8)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[contour == 255] = [255, 0, 0]  # Red contour
    return overlay

def process_all(base_dir, output_dir, shape=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    brain_dir = os.path.join(base_dir, "Task1", "brain")
    img_id = 0

    for patient in os.listdir(brain_dir):
        if patient == "overview":
            continue
        paths = {
            "ct": os.path.join(brain_dir, patient, "ct.nii.gz"),
            "mr": os.path.join(brain_dir, patient, "mr.nii.gz"),
            "mask": os.path.join(brain_dir, patient, "mask.nii.gz"),
        }
        if not all(os.path.exists(p) for p in paths.values()):
            continue

        ct = normalize(nib.load(paths["ct"]).get_fdata(), -1000, 1000)
        mr = normalize(nib.load(paths["mr"]).get_fdata())
        mask = normalize(nib.load(paths["mask"]).get_fdata())

        for i in range(ct.shape[2]):
            ct_slice = resize(ct[:, :, i], shape, preserve_range=True)
            mr_slice = resize(mr[:, :, i], shape, preserve_range=True)
            mask_slice = resize(mask[:, :, i], shape, preserve_range=True)

            if np.max(mask_slice) < 0.1:
                continue  # skip blank masks

            contour = generate_contour(mask_slice)
            if contour is None:
                continue

            ct_with_contour = overlay_image_with_contour(ct_slice, contour)
            mr_with_contour = overlay_image_with_contour(mr_slice, contour)

            Image.fromarray(ct_with_contour).save(os.path.join(output_dir, f"{img_id:05d}_ctcontour.png"))
            Image.fromarray(mr_with_contour).save(os.path.join(output_dir, f"{img_id:05d}_mrcontour.png"))
            img_id += 1


if __name__ == "__main__":
    process_all(".", "PreprocessedData", shape=(256, 256))