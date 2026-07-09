"""
Regenerates ChronoDepth visualization videos from the raw .npy predictions,
using the inferno colormap for consistency with VDA and DepthCrafter.

Usage:
    python colorize_chronodepth.py --npy_path /content/chrono_outputs/tum_freiburg1_desk_depth.npy \
                                    --output_path /content/chrono_outputs/tum_freiburg1_desk_depth_recolored.mp4

Or batch mode over all sequences + qualitative video:
    python colorize_chronodepth.py --all --chrono_output_dir /content/chrono_outputs
"""
import argparse
import os

import cv2
import matplotlib.cm as cm
import numpy as np

TUM_TAGS = ["freiburg1_desk", "freiburg2_desk", "freiburg3_office"]
QUALITATIVE_TAG = "bumba_meu_boi"


def equalize_frame(frame):
    flat = frame.flatten()
    ranks = np.argsort(np.argsort(flat))
    return (ranks / (len(flat) - 1)).reshape(frame.shape)


def colorize_frame(frame, cmap_name="inferno"):
    equalized = equalize_frame(frame)
    inverted = 1 - equalized
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(inverted)[:, :, :3]
    return (colored * 255).astype(np.uint8)


def build_video(frames_rgb, output_path, fps=10):
    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def colorize_npy(npy_path, output_path, fps=10):
    arr = np.load(npy_path)
    frames = [colorize_frame(arr[i]) for i in range(arr.shape[0])]
    build_video(frames, output_path, fps)
    print(f"Saved {output_path}")


def run_all(chrono_output_dir, fps=10):
    for tag in TUM_TAGS:
        npy_path = os.path.join(chrono_output_dir, f"tum_{tag}_depth.npy")
        if os.path.exists(npy_path):
            out_path = os.path.join(chrono_output_dir, f"tum_{tag}_depth_recolored.mp4")
            colorize_npy(npy_path, out_path, fps)

    qual_npy = os.path.join(chrono_output_dir, "qualitative", f"{QUALITATIVE_TAG}_depth.npy")
    if os.path.exists(qual_npy):
        out_path = os.path.join(chrono_output_dir, "qualitative", f"{QUALITATIVE_TAG}_depth_recolored.mp4")
        colorize_npy(qual_npy, out_path, fps=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--chrono_output_dir", default="/content/chrono_outputs")
    parser.add_argument("--npy_path")
    parser.add_argument("--output_path")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    if args.all:
        run_all(args.chrono_output_dir, args.fps)
    else:
        if not args.npy_path or not args.output_path:
            parser.error("either --all or (--npy_path and --output_path) is required")
        colorize_npy(args.npy_path, args.output_path, args.fps)


if __name__ == "__main__":
    main()