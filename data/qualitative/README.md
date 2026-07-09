`bumba_meu_boi.mp4` is committed here directly, alongside the TUM data —
no separate upload or setup step needed. It's a short clip (~15s, ~467
frames at 30 fps) of *Bumba Meu Boi Raízes do Maranhão*, a Maranhão
cultural celebration, used for the qualitative-only case study in the
paper (no ground-truth depth available for this domain).

Every inference script (`infer_vda.py`, `infer_depthcrafter.py`,
`infer_chronodepth.py`) automatically detects this file and processes it
alongside the 3 TUM sequences when run with `--all`.
