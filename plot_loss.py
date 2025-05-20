import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
# List every log‐folder you want to include (each must contain
# its own events.out.tfevents.* files, either directly or in subfolders):
log_dirs = [
    '/home/sauravdosi/mediffuse/diffusers/examples/instruct_pix2pix/instruct-ct2mri/logs',
    '/home/sauravdosi/mediffuse/diffusers/examples/instruct_pix2pix/instruct-ct2mri-large/logs',
    '/home/sauravdosi/mediffuse/diffusers/examples/instruct_pix2pix/instruct-mri2ct-large/logs',
    '/home/sauravdosi/mediffuse/ControlNet/lightning_logs/version_0'
    # add more as you like…
]
legend_labels = [
    'Instruct Pix2Pix CT→MRI (small)',
    'Instruct Pix2Pix CT→MRI (large)',
    'Instruct Pix2Pix MRI→CT (large)',
    'ControlNet CT Contours→MRI',
]
# Candidate tags to try (in order) for each run:
candidate_tags   = ['train_loss', 'train/loss_step']

# Output filename:
output_png       = 'combined_train_loss_all.png'

# Smoothing window and zoom margin:
smoothing_window = 50
zoom_margin_frac = 0.05
# ───────────────────────────────────────────────────────────────────────────────

def find_event_files(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, '**', 'events.out.tfevents.*'),
                              recursive=True))

def extract_loss(event_file, tag):
    for rec in tf.compat.v1.train.summary_iterator(event_file):
        if not rec.summary: continue
        for v in rec.summary.value:
            if v.tag == tag:
                yield rec.step, v.simple_value

def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, mode='valid')

def plot_multiple_runs(log_dirs, candidate_tags, output_path):
    plt.figure(figsize=(6,4))
    all_smoothed = []

    for idx, run_dir in enumerate(log_dirs):
        run_name = os.path.basename(run_dir.rstrip('/'))
        # find which tag works for this run
        chosen_tag = None
        points = []
        for tag in candidate_tags:
            # try extracting a few points
            pts = []
            for ef in find_event_files(run_dir):
                pts.extend(extract_loss(ef, tag))
                if pts:
                    break
            if pts:
                chosen_tag = tag
                points = pts  # keep the initial ones
                break

        if not chosen_tag:
            print(f"⚠ No matching tags in {run_name}; tried {candidate_tags}")
            continue
        print(f"→ Run '{run_name}' using tag '{chosen_tag}'")

        # now extract all for that tag
        for ef in find_event_files(run_dir):
            points.extend(extract_loss(ef, chosen_tag))

        # sort & unzip
        points = sorted(points, key=lambda x: x[0])
        steps = np.array([p[0] for p in points])
        losses = np.array([p[1] for p in points])

        # smooth
        if len(losses) >= smoothing_window:
            sm_losses = moving_average(losses, smoothing_window)
            w = smoothing_window
            offset_start = (w - 1) // 2
            offset_end   = (w - 1) - offset_start
            sm_steps = steps[offset_start:len(steps)-offset_end]
        else:
            print(f"⚠ Not enough points to smooth for {run_name}")
            sm_steps, sm_losses = steps, losses

        all_smoothed.append(sm_losses)
        plt.plot(sm_steps,
                 sm_losses,
                 label=legend_labels[idx])

    # auto-zoom
    if all_smoothed:
        mn = min(arr.min() for arr in all_smoothed)
        mx = max(arr.max() for arr in all_smoothed)
        m = (mx - mn) * zoom_margin_frac
        plt.ylim(mn - m, mx + m)

    plt.xlabel('Training Step')
    plt.ylabel('Smoothed Loss')
    plt.title(f'Combined Training Loss (MA window={smoothing_window})')
    plt.legend(fontsize='small', ncol=1)
    plt.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved combined plot as {output_path}")

if __name__ == '__main__':
    plot_multiple_runs(log_dirs, candidate_tags, output_png)