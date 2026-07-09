"""
[ADVANCED / OPTIONAL] Runs ChronoDepth inference on the prepared TUM
videos and, if present, the qualitative Bumba Meu Boi video.

Regenerates the predictions already committed under predictions/chronodepth/.
Most people reproducing the paper's numbers don't need this -- see the
README's "Quick reproduction" section. Needs a GPU.

Setup (once per GPU session):
    git clone https://github.com/jhaoshao/ChronoDepth
    cd ChronoDepth
    pip install diffusers==0.29.1 einops==0.8.0 matplotlib==3.8.4 \
                mediapy==1.2.2 tqdm==4.66.2 opencv-python
    pip install transformers==4.43.3

transformers must be installed after the base requirements, or
run_infer.py fails on the first run.

Usage (from inside the ChronoDepth directory):
    python ../scripts/infer_chronodepth.py --all --repo_root ..

Also runs colorize_chronodepth.py at the end, so the visualization videos
match the ones used in the qualitative comparison figure.

Or a single video:
    python ../scripts/infer_chronodepth.py --data_dir ../data/processed/tum_freiburg1_desk.mp4 \
                                            --output_dir ../predictions/chronodepth
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


def run_chronodepth(data_dir, output_dir, denoise_steps=5, chunk_size=5, n_tokens=10):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "python", "run_infer.py",
        "--unet=jhshao/ChronoDepth-v1",
        "--model_base=stabilityai/stable-video-diffusion-img2vid-xt",
        "--seed=1234",
        f"--data_dir={data_dir}",
        f"--output_dir={output_dir}",
        f"--denoise_steps={denoise_steps}",
        f"--chunk_size={chunk_size}",
        f"--n_tokens={n_tokens}",
        "--save_npy",
    ]
    subprocess.run(cmd, check=True)


def shrink_npy_files(output_dir):
    """Downcast to float16 before committing, keeping file sizes safely
    under GitHub's 100MB limit."""
    for f in glob.glob(os.path.join(output_dir, "*.npy")):
        arr = np.load(f)
        np.save(f, arr.astype(np.float16))
        print(f"[shrink] {f} -> float16")


def run_all(repo_root, output_dir=None):
    output_dir = output_dir or os.path.join(repo_root, "predictions", "chronodepth")
    processed_dir = os.path.join(repo_root, "data", "processed")
    for tag, fname in TUM_VIDEOS.items():
        video_path = os.path.join(processed_dir, fname)
        print(f"--- ChronoDepth: {tag} ---")
        run_chronodepth(video_path, output_dir)
    shrink_npy_files(output_dir)

    qualitative_path = os.path.join(repo_root, "data", "qualitative", QUALITATIVE_VIDEO)
    if os.path.exists(qualitative_path):
        print("--- ChronoDepth: qualitative (Bumba Meu Boi) ---")
        qualitative_output = os.path.join(output_dir, "qualitative")
        run_chronodepth(qualitative_path, qualitative_output)
    else:
        print(f"[skip] qualitative video not found at {qualitative_path}")

    colorize_script = os.path.join(repo_root, "scripts", "colorize_chronodepth.py")
    subprocess.run(["python", colorize_script, "--all", "--chrono_output_dir", output_dir], check=True)

    qualitative_output = os.path.join(output_dir, "qualitative")
    for f in glob.glob(os.path.join(qualitative_output, "*.npy")):
        os.remove(f)
        print(f"[skip commit] removed {f} (only *_recolored.mp4 is needed for the qualitative figure)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="run on all TUM sequences + qualitative video")
    parser.add_argument("--repo_root", default="..", help="path to the repo root (for --all mode)")
    parser.add_argument("--data_dir", help="single-video mode: path to input video")
    parser.add_argument("--output_dir", default=None, help="defaults to <repo_root>/predictions/chronodepth")
    parser.add_argument("--denoise_steps", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=5)
    parser.add_argument("--n_tokens", type=int, default=10)
    args = parser.parse_args()

    if args.all:
        run_all(args.repo_root, args.output_dir)
    else:
        if not args.data_dir:
            parser.error("either --all or --data_dir is required")
        output_dir = args.output_dir or os.path.join(args.repo_root, "predictions", "chronodepth")
        run_chronodepth(args.data_dir, output_dir, args.denoise_steps, args.chunk_size, args.n_tokens)


if __name__ == "__main__":
    main()