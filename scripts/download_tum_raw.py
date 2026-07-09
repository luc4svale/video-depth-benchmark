"""
Downloads the raw TUM RGB-D sequences and extracts them.

Only needed before running scripts/eval_metrics.py or
notebooks/05_evaluate_and_plot.ipynb -- the ground-truth depth maps aren't
committed to the repo, only the processed videos are.

Usage:
    python download_tum_raw.py --output_dir /content/tum_raw
"""
import argparse
import os
import subprocess

TUM_SEQUENCES = {
    "fr1": {
        "name": "rgbd_dataset_freiburg1_desk",
        "url": "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
    },
    "fr2": {
        "name": "rgbd_dataset_freiburg2_desk",
        "url": "https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz",
    },
    "fr3": {
        "name": "rgbd_dataset_freiburg3_long_office_household",
        "url": "https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz",
    },
}


def download_and_extract(info, output_dir):
    seq_path = os.path.join(output_dir, info["name"])
    if os.path.exists(seq_path):
        print(f"[skip] already extracted: {seq_path}")
        return seq_path
    os.makedirs(output_dir, exist_ok=True)
    tgz_path = os.path.join(output_dir, f"{info['name']}.tgz")
    subprocess.run(["wget", "-q", info["url"], "-O", tgz_path, "--no-check-certificate"], check=True)
    subprocess.run(["tar", "-xzf", tgz_path, "-C", output_dir], check=True)
    print(f"[ok] extracted: {seq_path}")
    return seq_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/content/tum_raw")
    args = parser.parse_args()
    for info in TUM_SEQUENCES.values():
        download_and_extract(info, args.output_dir)


if __name__ == "__main__":
    main()