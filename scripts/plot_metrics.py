"""
Generates the bar chart comparing average metrics across the three models,
reading the MEDIA rows from results/all_models_metrics_consolidated.csv.

Usage:
    python plot_metrics.py --results_dir results
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt

MODEL_ORDER = ["VideoDepthAnything-vits", "DepthCrafter", "ChronoDepth"]
MODEL_LABELS = {"VideoDepthAnything-vits": "VDA (vits)", "DepthCrafter": "DepthCrafter", "ChronoDepth": "ChronoDepth"}
COLORS = ["#2a78d6", "#1baf7a", "#eda100"]


def load_averages(csv_path):
    averages = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["sequence"] == "MEDIA":
                averages[row["model"]] = {
                    "AbsRel": float(row["AbsRel"]),
                    "RMSE": float(row["RMSE"]),
                    "delta1": float(row["delta1"]),
                    "TAE": float(row["TAE"]),
                }
    return averages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "all_models_metrics_consolidated.csv")
    averages = load_averages(csv_path)

    models = [m for m in MODEL_ORDER if m in averages]
    labels = [MODEL_LABELS[m] for m in models]

    metrics = {
        r"AbsRel ($\downarrow$)": [averages[m]["AbsRel"] for m in models],
        r"RMSE ($\downarrow$)": [averages[m]["RMSE"] for m in models],
        r"$\delta_1$ ($\uparrow$)": [averages[m]["delta1"] for m in models],
        r"TAE ($\downarrow$)": [averages[m]["TAE"] for m in models],
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(labels, values, color=COLORS[:len(labels)], width=0.6)
        ax.set_title(metric_name, fontsize=12)
        ax.set_ylim(0, max(values) * 1.25)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Average metrics comparison on TUM RGB-D", fontsize=13, y=1.05)
    plt.tight_layout()

    png_path = os.path.join(args.results_dir, "comparacao_metricas_tum.png")
    pdf_path = os.path.join(args.results_dir, "comparacao_metricas_tum.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()