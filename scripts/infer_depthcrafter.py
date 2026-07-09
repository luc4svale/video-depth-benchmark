"""
[ADVANCED / OPTIONAL] Runs DepthCrafter inference on the prepared TUM
videos and, if present, the qualitative Bumba Meu Boi video.

Regenerates the predictions already committed under predictions/depthcrafter/.
Most people reproducing the paper's numbers don't need this -- see the
README's "Quick reproduction" section. Needs a GPU.

Setup (once per GPU session):
    git clone https://github.com/Tencent/DepthCrafter.git
    cd DepthCrafter
    pip install accelerate diffusers fire gradio huggingface-hub matplotlib mediapy transformers eva-decord

DepthCrafter's pyproject.toml requires Python >=3.13, so "pip install -e ."
fails on Colab (Python 3.12). Not needed anyway, run.py runs fine from
inside this directory without an editable install.

Usage (from inside the DepthCrafter directory):
    python ../scripts/infer_depthcrafter.py --all --repo_root ..

DepthCrafter's run.py always writes into DepthCrafter/demo_output/, so
this script copies the results into predictions/depthcrafter/ afterwards.

Or a single video:
    python ../scripts/infer_depthcrafter.py --video_path ../data/processed/tum_freiburg1_desk.mp4 \
                                             --repo_root ..
"""
import argparse
import glob
import os
import shutil
import subprocess

import numpy as np

TUM_VIDEOS = {
    "freiburg1_desk": "tum_freiburg1_desk.mp4",
    "freiburg2_desk": "tum_freiburg2_desk.mp4",
    "freiburg3_office": "tum_freiburg3_office.mp4",
}
QUALITATIVE_VIDEO = "bumba_meu_boi.mp4"
DEMO_OUTPUT_DIR = "demo_output"  # DepthCrafter's own repo, fixed by upstream run.py


def run_depthcrafter(video_path, max_res=512, save_npz=True):
    cmd = ["python", "run.py", "--video-path", video_path, "--max-res", str(max_res)]
    if save_npz:
        cmd.append("--save-npz")
    subprocess.run(cmd, check=True)


def collect_outputs(predictions_dir, subdir=None):
    """Copies the .npz and _vis.mp4 from demo_output/ into predictions/,
    then clears demo_output/ so the next video doesn't mix in."""
    dest = os.path.join(predictions_dir, subdir) if subdir else predictions_dir
    os.makedirs(dest, exist_ok=True)
    for pattern in ("*.npz", "*_vis.mp4"):
        for f in glob.glob(os.path.join(DEMO_OUTPUT_DIR, pattern)):
            shutil.copy(f, dest)
    for f in glob.glob(os.path.join(DEMO_OUTPUT_DIR, "*")):
        os.remove(f) if os.path.isfile(f) else shutil.rmtree(f)


def shrink_npz_files(output_dir):
    """Downcast to float16 before committing, keeping file sizes safely
    under GitHub's 100MB limit."""
    for f in glob.glob(os.path.join(output_dir, "*.npz")):
        data = np.load(f)
        arrays = {k: v.astype(np.float16) for k, v in data.items()}
        np.savez_compressed(f, **arrays)
        print(f"[shrink] {f} -> float16")


def run_all(repo_root, predictions_dir=None, max_res=512):
    predictions_dir = predictions_dir or os.path.join(repo_root, "predictions", "depthcrafter")
    processed_dir = os.path.join(repo_root, "data", "processed")
    for tag, fname in TUM_VIDEOS.items():
        video_path = os.path.join(processed_dir, fname)
        print(f"--- DepthCrafter: {tag} ---")
        run_depthcrafter(video_path, max_res)
        collect_outputs(predictions_dir)
    shrink_npz_files(predictions_dir)

    qualitative_path = os.path.join(repo_root, "data", "qualitative", QUALITATIVE_VIDEO)
    if os.path.exists(qualitative_path):
        print("--- DepthCrafter: qualitative (Bumba Meu Boi) ---")
        qualitative_output = os.path.join(predictions_dir, "qualitative")
        run_depthcrafter(qualitative_path, max_res)
        collect_outputs(predictions_dir, subdir="qualitative")
        for f in glob.glob(os.path.join(qualitative_output, "*.npz")):
            os.remove(f)
            print(f"[skip commit] removed {f} (only *_vis.mp4 is needed for the qualitative figure)")
    else:
        print(f"[skip] qualitative video not found at {qualitative_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="run on all TUM sequences + qualitative video")
    parser.add_argument("--repo_root", default="..", help="path to the repo root (for --all mode)")
    parser.add_argument("--video_path", help="single-video mode")
    parser.add_argument("--predictions_dir", default=None, help="defaults to <repo_root>/predictions/depthcrafter")
    parser.add_argument("--max_res", type=int, default=512)
    args = parser.parse_args()

    if args.all:
        run_all(args.repo_root, args.predictions_dir, args.max_res)
    else:
        if not args.video_path:
            parser.error("either --all or --video_path is required")
        run_depthcrafter(args.video_path, args.max_res)
        predictions_dir = args.predictions_dir or os.path.join(args.repo_root, "predictions", "depthcrafter")
        collect_outputs(predictions_dir)
        shrink_npz_files(predictions_dir)


if __name__ == "__main__":
    main()