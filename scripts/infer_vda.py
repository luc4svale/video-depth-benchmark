"""
[ADVANCED / OPTIONAL] Runs VDA (ViT-S) inference on the prepared TUM
videos and, if present, the qualitative Bumba Meu Boi video.

Regenerates the predictions already committed under predictions/vda/.
Most people reproducing the paper's numbers don't need this -- see the
README's "Quick reproduction" section. Needs a GPU.

Setup (once per GPU session):
    git clone https://github.com/DepthAnything/Video-Depth-Anything.git
    cd Video-Depth-Anything
    pip install -r requirements.txt
    pip install imageio==2.37.0 imageio-ffmpeg==0.4.7 einops easydict decord
    bash get_weights.sh

Usage (from inside the Video-Depth-Anything directory):
    python ../scripts/infer_vda.py --all --repo_root ..

Or a single video:
    python ../scripts/infer_vda.py --input_video ../data/processed/tum_freiburg1_desk.mp4 \
                                    --output_dir ../predictions/vda
"""
import argparse
import glob
import os
import subprocess

import numpy as np

TUM_VIDEOS = {
    "freiburg1_desk": "tum_freiburg1_desk.mp4",
    "freiburg2_desk": "tum_freiburg2_desk.mp4",
    "freiburg3_office": "tum_freiburg3_office.mp4",
}
QUALITATIVE_VIDEO = "bumba_meu_boi.mp4"


def run_vda(input_video, output_dir, encoder="vits", target_fps=10, save_npz=True):
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["python", "run.py", "--input_video", input_video, "--output_dir", output_dir, "--encoder", encoder]
    if target_fps is not None:
        cmd += ["--target_fps", str(target_fps)]
    if save_npz:
        cmd.append("--save_npz")
    subprocess.run(cmd, check=True)


def shrink_npz_files(output_dir):
    """VDA saves .npz as uncompressed float32, close to GitHub's 100MB
    limit. Downcasting to float16 halves the size, precision loss is
    irrelevant for these metrics."""
    for f in glob.glob(os.path.join(output_dir, "*.npz")):
        data = np.load(f)
        arrays = {k: v.astype(np.float16) for k, v in data.items()}
        np.savez_compressed(f, **arrays)
        print(f"[shrink] {f} -> float16")


def run_all(repo_root, output_dir=None, encoder="vits", target_fps=10):
    output_dir = output_dir or os.path.join(repo_root, "predictions", "vda")
    processed_dir = os.path.join(repo_root, "data", "processed")
    for tag, fname in TUM_VIDEOS.items():
        video_path = os.path.join(processed_dir, fname)
        print(f"--- VDA: {tag} ---")
        run_vda(video_path, output_dir, encoder, target_fps)
    shrink_npz_files(output_dir)

    qualitative_path = os.path.join(repo_root, "data", "qualitative", QUALITATIVE_VIDEO)
    if os.path.exists(qualitative_path):
        print("--- VDA: qualitative (Bumba Meu Boi) ---")
        qualitative_output = os.path.join(output_dir, "qualitative")
        run_vda(qualitative_path, qualitative_output, encoder, target_fps=None)
        for f in glob.glob(os.path.join(qualitative_output, "*.npz")):
            os.remove(f)
            print(f"[skip commit] removed {f} (only *_vis.mp4 is needed for the qualitative figure)")
    else:
        print(f"[skip] qualitative video not found at {qualitative_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="run on all TUM sequences + qualitative video")
    parser.add_argument("--repo_root", default="..", help="path to the repo root (for --all mode)")
    parser.add_argument("--input_video", help="single-video mode")
    parser.add_argument("--output_dir", default=None, help="defaults to <repo_root>/predictions/vda")
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitl"])
    parser.add_argument("--target_fps", type=int, default=10)
    args = parser.parse_args()

    if args.all:
        run_all(args.repo_root, args.output_dir, args.encoder, args.target_fps)
    else:
        if not args.input_video:
            parser.error("either --all or --input_video is required")
        run_vda(args.input_video, args.output_dir or os.path.join(args.repo_root, "predictions", "vda"),
                 args.encoder, args.target_fps)


if __name__ == "__main__":
    main()