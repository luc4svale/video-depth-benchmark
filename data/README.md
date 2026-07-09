# Data

## `processed/`

The 3 TUM RGB-D sequences, built once by `notebooks/01_prepare_tum_data.ipynb`:

```
processed/
  tum_freiburg1_desk.mp4      tum_freiburg1_desk_gt_depth.npz      tum_pairs_fr1.json
  tum_freiburg2_desk.mp4      tum_freiburg2_desk_gt_depth.npz      tum_pairs_fr2.json
  tum_freiburg3_office.mp4    tum_freiburg3_office_gt_depth.npz    tum_pairs_fr3.json
```

- `.mp4` — H.264, baseline profile, no B-frames (required by `decord`,
  used by VDA), 10fps, 110 frames — same protocol as DepthCrafter's paper.
- `_gt_depth.npz` — the 110 ground-truth depth maps, in meters. What
  `scripts/eval_metrics.py` compares predictions against. No raw TUM
  download needed to run evaluation.
- `tum_pairs_*.json` — which original TUM RGB/depth filenames each frame
  came from. Provenance only, not needed to run anything.

## `qualitative/`

`bumba_meu_boi.mp4`, used for the qualitative-only comparison. No ground
truth available for this domain.

## Raw TUM RGB-D

Not committed, only needed to regenerate `processed/`:

- `freiburg1_desk`: https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
- `freiburg2_desk`: https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
- `freiburg3_long_office_household`: https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz

Downloaded by `scripts/download_tum_raw.py`, called from
`scripts/prepare_tum_videos.py`.
