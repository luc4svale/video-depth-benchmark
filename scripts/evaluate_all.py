"""
Evaluates all 3 models on all 3 TUM sequences and writes the CSVs under
results/. Main entry point for reproducing the paper's Table I / Table II.

Reads everything already committed in the repo:
  - predictions/<model>/...
  - data/processed/tum_*_gt_depth.npz

No downloads, no GPU.

Usage (from the repo root):
    python scripts/evaluate_all.py
"""
import argparse
import csv
import os

import numpy as np

from eval_metrics import evaluate

SEQUENCES = ["freiburg1_desk", "freiburg2_desk", "freiburg3_office"]

MODEL_PRED_FILES = {
    "vda": lambda seq: f"tum_{seq}_depths.npz",
    "depthcrafter": lambda seq: f"tum_{seq}.npz",
    "chronodepth": lambda seq: f"tum_{seq}_depth.npy",
}

MODEL_DISPLAY_NAME = {
    "vda": "VideoDepthAnything-vits",
    "depthcrafter": "DepthCrafter",
    "chronodepth": "ChronoDepth",
}


def run_model(model, predictions_dir, processed_dir):
    rows = []
    for seq in SEQUENCES:
        pred_fname = MODEL_PRED_FILES[model](seq)
        pred_path = os.path.join(predictions_dir, model, pred_fname)
        gt_depth_path = os.path.join(processed_dir, f"tum_{seq}_gt_depth.npz")

        if not os.path.exists(pred_path):
            print(f"[skip] {model}/{seq}: prediction not found at {pred_path}")
            continue
        if not os.path.exists(gt_depth_path):
            print(f"[skip] {model}/{seq}: ground truth not found, run notebooks/01_prepare_tum_data.ipynb")
            continue

        metrics = evaluate(model, pred_path, gt_depth_path)
        rows.append([MODEL_DISPLAY_NAME[model], seq,
                     f"{metrics['AbsRel']:.4f}", f"{metrics['RMSE']:.4f}",
                     f"{metrics['delta1']:.4f}", f"{metrics['TAE']:.4f}"])
        print(f"[ok] {model}/{seq}: AbsRel={metrics['AbsRel']:.4f} "
              f"RMSE={metrics['RMSE']:.4f} delta1={metrics['delta1']:.4f} TAE={metrics['TAE']:.4f}")
    return rows


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def add_average_row(rows, model_display_name):
    cols = np.array([[float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows])
    avg = cols.mean(axis=0)
    return rows + [[model_display_name, "MEDIA", f"{avg[0]:.4f}", f"{avg[1]:.4f}",
                    f"{avg[2]:.4f}", f"{avg[3]:.4f}"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", default="predictions")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    header = ["model", "sequence", "AbsRel", "RMSE", "delta1", "TAE"]
    all_rows = []

    for model in ("vda", "depthcrafter", "chronodepth"):
        rows = run_model(model, args.predictions_dir, args.processed_dir)
        if not rows:
            continue
        rows_with_avg = add_average_row(rows, MODEL_DISPLAY_NAME[model])
        write_csv(os.path.join(args.results_dir, f"{model}_metrics.csv"), header, rows_with_avg)
        all_rows.extend(rows_with_avg)

    if all_rows:
        write_csv(os.path.join(args.results_dir, "all_models_metrics_consolidated.csv"), header, all_rows)
        print(f"\nWrote consolidated results to {args.results_dir}/all_models_metrics_consolidated.csv")
    else:
        print("\nNo predictions found. Check predictions/ (see predictions/README.md).")


if __name__ == "__main__":
    main()