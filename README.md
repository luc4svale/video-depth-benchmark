# Video Depth Anything — Reproduction and Evaluation on Out-of-Domain Videos

Reproduction of **Video Depth Anything (VDA)** [arXiv:2501.12375](https://arxiv.org/abs/2501.12375),
evaluated zero-shot against **DepthCrafter** and **ChronoDepth** — the same
two models the original paper compares against — on data outside VDA's
training domain: **TUM RGB-D** (real ground-truth depth) and a qualitative
video of *Bumba Meu Boi Raízes do Maranhão*, a Maranhão cultural
celebration with no reference depth.

Developed for the Digital Image Processing course, Computer Engineering
Program, Federal University of Maranhão (UFMA).

Paper: **"Video Depth Anything: Reproduction and Evaluation on Out-of-Domain
Videos"** — `paper/paper_en.pdf` (English), `paper/paper_pt.pdf` (Portuguese).

## Authors

- Antonio Lucas da Silva Vale
- Jeanderson da Silva Campos
- Lucas Martins Campos Matos ([lucas.mcm@discente.ufma.br](mailto:lucas.mcm@discente.ufma.br))

Advisor: Prof. Haroldo Gomes Barroso Filho (Computer Engineering
Coordination, UFMA)

## Prerequisites

- Python 3.10+
- `ffmpeg`, only needed if regenerating `data/processed/` from scratch:
```bash
sudo apt install ffmpeg      # Ubuntu/Debian
brew install ffmpeg          # macOS
sudo dnf install ffmpeg      # Fedora
sudo pacman -S ffmpeg        # Arch
```
- GPU, only needed to regenerate predictions (see below). Not needed to
  reproduce the paper's numbers.

## Quick reproduction

The TUM RGB-D data and the raw model predictions are already committed.
Reproducing the paper's tables and chart needs no GPU and no downloads:

```bash
git clone https://github.com/luc4svale/video-depth-benchmark.git
cd video-depth-benchmark
pip install -r requirements.txt

python scripts/evaluate_all.py     # -> results/*.csv
python scripts/plot_metrics.py     # -> results/comparacao_metricas_tum.{png,pdf}
```

`results/all_models_metrics_consolidated.csv` should now match Tables I
and II in the paper. `notebooks/05_evaluate_and_plot.ipynb` runs the same
steps if you prefer notebooks.

Regenerating the predictions themselves needs a GPU — see
[Regenerating predictions from scratch](#regenerating-predictions-from-scratch).

## Results

Average metrics over 3 TUM RGB-D sequences (`freiburg1_desk`,
`freiburg2_desk`, `freiburg3_long_office_household`), 110 frames each:

| Model         | AbsRel ↓ | RMSE ↓ | δ1 ↑   | TAE ↓  |
|---------------|---------|--------|--------|--------|
| VDA (ViT-S)   | 0.0901  | 0.4836 | 0.9464 | 0.0437 |
| DepthCrafter  | 0.0886  | 0.3462 | 0.9457 | 0.0382 |
| ChronoDepth   | 0.1342  | 0.4866 | 0.8411 | 0.0244 |

VDA and DepthCrafter are essentially equivalent in geometric accuracy.
ChronoDepth trades accuracy for temporal consistency: lowest TAE, but
AbsRel 49% higher than VDA's. This diverges from the original paper, where
VDA clearly outperforms DepthCrafter on KITTI, Sintel, ScanNet, NYUv2 and
Bonn. Two reasons: we use the ViT-S variant, lighter than the paper's
primary ViT-L model, and TUM RGB-D is outside both models' training data.
Full discussion in the paper, Section V.

Per-sequence results are in `results/all_models_metrics_consolidated.csv`.
Qualitative comparison (Figure 3) is in the paper.

A third model, **DepthAnyVideo**, was tested but excluded: it processes
the whole video in one pass with no windowing and ran out of memory on a
15GB GPU even at low resolution.

## Repository structure

```
data/
  processed/          TUM videos, ground-truth depth, RGB-depth pairing
  qualitative/         Bumba Meu Boi video

predictions/          Raw model outputs -- what evaluate_all.py reads
  vda/
  depthcrafter/
  chronodepth/

scripts/
  evaluate_all.py       Main entry point: predictions/ + data/processed/ -> results/*.csv
  plot_metrics.py        Regenerates the comparison chart
  eval_metrics.py         Per-sequence metric computation
  infer_vda.py            Regenerate predictions/vda/ (GPU)
  infer_depthcrafter.py   Regenerate predictions/depthcrafter/ (GPU)
  infer_chronodepth.py    Regenerate predictions/chronodepth/ (GPU)
  colorize_chronodepth.py Used by infer_chronodepth.py
  prepare_tum_videos.py   Regenerate data/processed/ from raw TUM RGB-D
  download_tum_raw.py     Used by prepare_tum_videos.py

notebooks/             One notebook per stage above
  01_prepare_tum_data.ipynb
  02_infer_vda.ipynb
  03_infer_depthcrafter.ipynb
  04_infer_chronodepth.ipynb
  05_evaluate_and_plot.ipynb   Main entry point

results/               Metrics CSVs and comparison chart
paper/                 Final paper, English and Portuguese
```

## Regenerating predictions from scratch

Only needed to re-run the models yourselves. Requires a GPU.

Running locally: clone the repo once, open the notebooks with the working
directory at the repo root — the first code cell checks and fixes this
automatically.

Running on Colab: every notebook has a cell that detects Colab and clones
the repo into the VM. Skipped when running locally. Everything after works
the same either way.

Order:

1. `01_prepare_tum_data.ipynb` — only if `data/processed/` is empty.
2. `02_infer_vda.ipynb`, `03_infer_depthcrafter.ipynb`,
   `04_infer_chronodepth.ipynb` — independent, run in any order.
3. `05_evaluate_and_plot.ipynb` (or `evaluate_all.py` + `plot_metrics.py`)
   — regenerates `results/`.

Commit the regenerated files after each step so future clones stay current.

## Disparity conventions

All three models predict relative disparity, not metric depth, and each
converts it differently — the easiest thing to get wrong here:

- **DepthCrafter:** disparity normalized to [0, 1] → `depth = 1 / pred`
- **VDA:** unnormalized disparity → `depth = 1 / pred`
- **ChronoDepth:** inverted → `depth = 1 / (1 - pred)`

The ChronoDepth convention was found empirically (paper, Section IV.D):
the direct formula gives implausible AbsRel, since pixels near disparity
1.0 map to very large depth instead of small. `eval_metrics.py` applies
the right formula based on the `--model` flag.

Depth is scale-aligned to ground truth via median ratio, and metrics are
computed only where both GT and prediction fall in [0.1m, 10m].

## Known pitfalls

- **decord needs B-frame-free videos.** Without `-profile:v baseline -bf 0`,
  `decord.VideoReader` (used by VDA) fails to open videos that play fine
  everywhere else.
- **DepthCrafter requires Python ≥3.13** in its `pyproject.toml`. `pip
  install -e .` fails on Colab (Python 3.12). Not needed anyway —
  `run.py` runs fine without an editable install.
- **`transformers==4.43.3` must be installed after ChronoDepth's own
  requirements**, or the first inference run fails.
- **VDA and DepthCrafter save uncompressed `.npz`**, close to or over
  GitHub's 100MB limit. `infer_vda.py` and `infer_depthcrafter.py`
  downcast to float16 automatically before committing.
- **TAE is simplified.** The paper's formula (Eq. 16) warps frames via
  optical flow before differencing. This implementation uses a direct
  frame-to-frame difference instead — reasonable given the fixed 10fps
  re-encoding, but not a literal match to Eq. 16.
- **DepthAnyVideo** was tested and discarded: no windowing, consistently
  out of memory on a 15GB GPU.

## Datasets

- **TUM RGB-D** [Sturm et al., IROS 2012]: real ground-truth depth.
  `freiburg1_desk` and `freiburg2_desk` are handheld desk scenes;
  `freiburg3_long_office_household` covers a larger office. 110
  frames/sequence, matching DepthCrafter's original protocol.
- **Bumba Meu Boi Raízes do Maranhão**: qualitative only, no ground truth.
  Moving crowds, artificial nighttime lighting, complex costume textures.

## Compared models

- **Video Depth Anything (VDA)** — ViT-S, official weights, relative depth mode.
  https://github.com/DepthAnything/Video-Depth-Anything
- **DepthCrafter** — https://github.com/Tencent/DepthCrafter
- **ChronoDepth** — https://github.com/jhaoshao/ChronoDepth

## Paper

Full mathematical formulation, architecture, methodology, and results are
in `paper/paper_en.pdf` and `paper/paper_pt.pdf`.
