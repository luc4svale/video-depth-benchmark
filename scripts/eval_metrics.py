"""
Evaluates one model/sequence prediction against TUM RGB-D ground truth:
AbsRel, RMSE, delta1, TAE.

Disparity conventions differ per model. All predictions are relative
disparity, not metric depth, and need inverting before comparing to GT:

    DepthCrafter: normalized disparity [0, 1] -> depth = 1 / pred
    VDA (vits):   unnormalized disparity      -> depth = 1 / pred
    ChronoDepth:  inverted convention         -> depth = 1 / (1 - pred)

The ChronoDepth convention was found empirically: the direct formula gives
implausible AbsRel, since pixels near disparity 1.0 map to very large
depth instead of small. Inverting via 1 - pred before the 1/x conversion
fixes it.

Alignment is scale-only, via median ratio between GT and predicted
disparity (Eq. 12 in the paper). Validity mask: 0.1m < depth < 10m.

TAE here is simplified: a direct frame-to-frame difference, without the
optical-flow warp from the paper's Eq. 16. Reasonable given the TUM
sequences were re-encoded at a fixed 10fps with limited inter-frame
motion, but won't match an optical-flow reimplementation exactly.

Usage:
    python eval_metrics.py \
        --model chronodepth \
        --pred_path predictions/chronodepth/tum_freiburg1_desk_depth.npy \
        --gt_depth_path data/processed/tum_freiburg1_desk_gt_depth.npz \
        --sequence_name freiburg1_desk
"""
import argparse

import cv2
import numpy as np

DEPTH_MIN, DEPTH_MAX = 0.1, 10.0


def load_gt_depths(gt_depth_path):
    return np.load(gt_depth_path)["depth"]


def load_pred(pred_path):
    """Always returns float32, regardless of the precision it was stored
    in (VDA predictions are float16 to stay under GitHub's 100MB limit)."""
    if pred_path.endswith(".npz"):
        data = np.load(pred_path)
        key = "depth" if "depth" in data.files else "depths"
        return data[key].astype(np.float32)
    return np.load(pred_path).astype(np.float32)


def resize_pred(pred, target_hw):
    H, W = target_hw
    if pred.shape[1:] == (H, W):
        return pred
    return np.stack([cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) for p in pred])


def to_disparity_convention(pred, model):
    model = model.lower()
    if model in ("vda", "depthcrafter"):
        return pred
    elif model == "chronodepth":
        return 1.0 - pred
    else:
        raise ValueError(f"Unknown model: {model}")


def compute_metrics(gt_depths, pred_raw, model):
    mask = (gt_depths > DEPTH_MIN) & (gt_depths < DEPTH_MAX)

    pred_disp = to_disparity_convention(pred_raw, model)

    gt_disp = np.zeros_like(gt_depths)
    gt_disp[mask] = 1.0 / gt_depths[mask]

    scale = np.median(gt_disp[mask]) / np.median(pred_disp[mask])
    pred_disp_aligned = pred_disp * scale

    pred_depth = np.zeros_like(pred_disp_aligned)
    valid_pred = pred_disp_aligned > 1e-6
    pred_depth[valid_pred] = 1.0 / pred_disp_aligned[valid_pred]

    sane_pred = (pred_depth > DEPTH_MIN) & (pred_depth < DEPTH_MAX)
    combined_mask = mask & valid_pred & sane_pred

    gt_masked = gt_depths[combined_mask]
    pred_masked = pred_depth[combined_mask]

    absrel = float(np.mean(np.abs(gt_masked - pred_masked) / gt_masked))
    rmse = float(np.sqrt(np.mean((gt_masked - pred_masked) ** 2)))
    thresh = np.maximum(gt_masked / pred_masked, pred_masked / gt_masked)
    delta1 = float(np.mean(thresh < 1.25))

    diffs = np.abs(np.diff(pred_depth, axis=0))
    diffs_masked = diffs[combined_mask[:-1] & combined_mask[1:]]
    tae = float(np.mean(diffs_masked))

    return {
        "scale": float(scale),
        "AbsRel": absrel,
        "RMSE": rmse,
        "delta1": delta1,
        "TAE": tae,
        "valid_pixels": int(combined_mask.sum()),
        "mask_pixels": int(mask.sum()),
    }


def evaluate(model, pred_path, gt_depth_path):
    gt_depths = load_gt_depths(gt_depth_path)
    pred = load_pred(pred_path)
    pred = resize_pred(pred, gt_depths.shape[1:])
    return compute_metrics(gt_depths, pred, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["vda", "depthcrafter", "chronodepth"])
    parser.add_argument("--pred_path", required=True, help=".npy or .npz raw prediction file")
    parser.add_argument("--gt_depth_path", required=True, help="tum_*_gt_depth.npz from data/processed/")
    parser.add_argument("--sequence_name", required=True)
    args = parser.parse_args()

    metrics = evaluate(args.model, args.pred_path, args.gt_depth_path)

    print(f"Model: {args.model} | Sequence: {args.sequence_name}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()