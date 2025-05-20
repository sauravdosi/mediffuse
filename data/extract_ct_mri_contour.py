import os
import cv2
import numpy as np
from PIL import Image

def normalize(image, vmin=None, vmax=None):
    """Optional clip and scale image to [0–1]."""
    img = image.astype(np.float32)
    if vmin is not None and vmax is not None:
        img = np.clip(img, vmin, vmax)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def generate_canny_edges(image, low_thresh=50, high_thresh=150):
    """Run Canny on a uint8 [0–255] image."""
    return cv2.Canny(image, low_thresh, high_thresh)

def extract_and_draw_contours(edges):
    """
    Find contours in a binary edge map and draw them on a blank BGR canvas.
    """
    # findContours modifies input, so copy
    edges_copy = edges.copy()
    contours, _ = cv2.findContours(edges_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create blank BGR canvas
    canvas = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)

    # draw contours in white, thickness=1
    cv2.drawContours(canvas, contours, -1, (255,255,255), 1)
    return canvas

def process_png(input_png_path, output_contour_path,
                low_thresh=50, high_thresh=150,
                normalize_vmin=None, normalize_vmax=None):
    # load as grayscale
    pil = Image.open(input_png_path).convert('L')
    img = np.array(pil)

    # optional normalize
    if normalize_vmin is not None or normalize_vmax is not None:
        img = normalize(img, normalize_vmin, normalize_vmax)
        img = (img * 255).astype(np.uint8)

    # get edges
    edges = generate_canny_edges(img, low_thresh, high_thresh)

    # draw contours on blank canvas
    contours_img = extract_and_draw_contours(edges)

    # save result
    os.makedirs(os.path.dirname(output_contour_path), exist_ok=True)
    cv2.imwrite(output_contour_path, contours_img)
    print(f"Saved contour-only image to {output_contour_path}")

if __name__ == "__main__":
    # example usage
    input_png  = "/home/sauravdosi/mediffuse/data/raw/CT/0042_CT.png"
    output_png = "/home/sauravdosi/mediffuse/presentation/0042_CT_contours.png"
    process_png(
        input_png,
        output_png,
        low_thresh=100,
        high_thresh=200,
        normalize_vmin=0,
        normalize_vmax=255
    )
