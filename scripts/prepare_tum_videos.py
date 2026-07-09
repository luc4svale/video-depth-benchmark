"""
Prepares the TUM RGB-D sequences: downloads them, pairs RGB and depth
frames by timestamp, and produces per sequence an H.264 video (no
B-frames, required by decord), a compressed .npz with ground-truth
depth in meters, and a .json recording the frame pairing.

Outputs are already committed under data/processed/, so most people
don't need to run this. Only needed to regenerate that folder from
scratch.

Usage:
    python prepare_tum_videos.py --raw_dir /tmp/tum_raw --output_dir data/processed
"""
import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image

from download_tum_raw import TUM_SEQUENCES as _TUM_SEQUENCES_BASE
from download_tum_raw import download_and_extract

_VIDEO_NAMES = {
    "fr1": "tum_freiburg1_desk.mp4",
    "fr2": "tum_freiburg2_desk.mp4",
    "fr3": "tum_freiburg3_office.mp4",
}
_GT_DEPTH_NAMES = {
    "fr1": "tum_freiburg1_desk_gt_depth.npz",
    "fr2": "tum_freiburg2_desk_gt_depth.npz",
    "fr3": "tum_freiburg3_office_gt_depth.npz",
}
TUM_SEQUENCES = {
    tag: {**info, "video_name": _VIDEO_NAMES[tag], "gt_depth_name": _GT_DEPTH_NAMES[tag]}
    for tag, info in _TUM_SEQUENCES_BASE.items()
}

MAX_FRAMES = 110  # DepthCrafter's original protocol
MAX_DIFF = 0.02   # max timestamp gap for RGB-depth pairing, in seconds
FPS = 10
DEPTH_SCALE = 5000.0  # TUM convention: raw uint16 PNG / 5000 = meters


def read_file_list(path):
    """Parses TUM's rgb.txt / depth.txt: 'timestamp filepath' per line."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            entries.append((float(parts[0]), parts[1]))
    return entries


def associate_pairs(seq_path, max_frames=MAX_FRAMES, max_diff=MAX_DIFF):
    """Greedy nearest-timestamp association between RGB and depth frames."""
    rgb_list = read_file_list(os.path.join(seq_path, "rgb.txt"))
    depth_list = read_file_list(os.path.join(seq_path, "depth.txt"))

    pairs = []
    depth_used = set()
    for t_rgb, f_rgb in rgb_list:
        best, best_diff = None, max_diff
        for i, (t_depth, _) in enumerate(depth_list):
            if i in depth_used:
                continue
            diff = abs(t_rgb - t_depth)
            if diff < best_diff:
                best, best_diff = i, diff
        if best is not None:
            depth_used.add(best)
            pairs.append((f_rgb, depth_list[best][1]))
        if len(pairs) >= max_frames:
            break
    return pairs


def build_video(frame_dir, output_path, fps=FPS):
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-profile:v", "baseline", "-bf", "0",
        "-pix_fmt", "yuv420p", output_path,
    ], check=True)


def save_gt_depth(seq_path, pairs, output_path):
    depths = []
    for _, f_depth in pairs:
        d = np.array(Image.open(os.path.join(seq_path, f_depth))).astype(np.float32) / DEPTH_SCALE
        depths.append(d)
    depths = np.stack(depths)
    np.savez_compressed(output_path, depth=depths)


def process_scene(tag, info, raw_dir, tmp_dir, output_dir):
    seq_path = download_and_extract(info, raw_dir)
    pairs = associate_pairs(seq_path)

    frame_dir = os.path.join(tmp_dir, f"tum_frames_{tag}")
    os.makedirs(frame_dir, exist_ok=True)
    for i, (f_rgb, _) in enumerate(pairs):
        src = os.path.join(seq_path, f_rgb)
        dst = os.path.join(frame_dir, f"frame_{i:04d}.png")
        subprocess.run(["cp", src, dst], check=True)

    os.makedirs(output_dir, exist_ok=True)
    pairs_json = os.path.join(output_dir, f"tum_pairs_{tag}.json")
    with open(pairs_json, "w") as f:
        json.dump(pairs, f)

    video_path = os.path.join(output_dir, info["video_name"])
    build_video(frame_dir, video_path)

    gt_depth_path = os.path.join(output_dir, info["gt_depth_name"])
    save_gt_depth(seq_path, pairs, gt_depth_path)

    print(f"[{tag}] {len(pairs)} pairs associated -> {video_path}, {gt_depth_path}")
    return video_path, gt_depth_path, pairs_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="/content/tum_raw")
    parser.add_argument("--tmp_dir", default="/content/tum_frames_tmp")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    for tag, info in TUM_SEQUENCES.items():
        process_scene(tag, info, args.raw_dir, args.tmp_dir, args.output_dir)

    print("\nDone. Commit the contents of", args.output_dir, "to data/processed/ in the repo.")


if __name__ == "__main__":
    main()