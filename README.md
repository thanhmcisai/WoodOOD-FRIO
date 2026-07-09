# WoodOOD-FRIO

Code for the paper **"Distance-Based OOD Detection is Brittle to Frequency
Degradation: A Fine-Grained Wood Benchmark and a Frequency-Robust Remedy."**

Out-of-distribution (OOD) detection for fine-grained wood-species identification
looks solved on clean images (distance/subspace detectors reach ~99 AUROC), but
that performance collapses under mild, deployment-realistic image degradation.
**FRIO** restores robustness by treating the brittleness as a *frequency*
phenomenon. Its main contribution is **FASC** (Frequency-Aware Score
Calibration), an augmentation-free, backbone-agnostic **test-time** correction of
the OOD score by an image's high-frequency energy ratio, complemented by two
optional training-time regularizers: **FCR** (Frequency-Consistency
Regularization) and **FOE** (Frequency-Outlier Exposure).

## Dataset

**WoodOOD-50** — 50 species, 18,598 macroscopic images, 17 botanical families —
is released separately:
<https://ptiteduvn-my.sharepoint.com/:u:/g/personal/khanhnt_ptit_edu_vn/EQkPywXle45Mt_vyE_bjwn8BfhgawzLyO9nrV3djhr4juQ?e=9GNhSD>

After downloading, set `paths.root` and `paths.data` in `configs/config.yaml`.

## Installation

Python ≥ 3.10 with a CUDA-enabled PyTorch (developed on Google Colab, L4 24 GB).

```bash
pip install -r requirements.txt
# torch / torchvision are assumed preinstalled (matching your CUDA); see requirements.txt
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/freq.py` | high-frequency energy ratio `r(x)` and band split — the core of FASC / FCR / FOE |
| `src/corruption.py` | ImageNet-C-style corruption suite + mild views (used for evaluation and FCR) |
| `src/isomax.py` | IsoMax+ head (prior work: Macêdo & Ludermir, 2021) |
| `src/eool.py` | IsoMax+ classifier + energy-based OOD regularization (basis for FOE) |
| `src/ood_eval.py` | OOD evaluation on real ID vs. near/far OOD (AUROC / FPR@95 / AUPR) |
| `src/embed_ood.py` | embedding-space OOD detection (distance / discriminator baselines) |
| `src/models.py` | four ImageNet-pretrained backbones + swappable CE / IsoMax+ head |
| `src/engine.py` | resumable ID-classification training loop (early stop on val macro-F1) |
| `src/data.py` | specimen-aware dataset / dataloaders from the manifest |
| `src/metrics.py`, `src/utils.py` | metrics, config/seed/paths/device helpers |
| `src/latent_space.py`, `src/sampling.py`, `src/generate_ood.py`, `src/sd_*.py`, `src/synth_quality.py` | diffusion-based outlier-synthesis pipeline (reported as a **negative result**) |
| `src/plotting.py`, `src/tables.py`, `src/gradcam.py` | figures/tables from results, Grad-CAM |
| `phase0_tools/` | dataset preparation: taxonomy build, pseudo-specimen clustering, specimen-aware splits |
| `configs/config.yaml` | single source of truth for all parameters |
| `configs/prompts.yaml` | text prompts for the synthesis pipeline |
| `NOTEBOOK_MAP.md` | registry of the driving notebook cells (pipeline order and outputs) |

All parameters are read from `configs/config.yaml`; nothing is hard-coded.

## Pipeline

1. **Data preparation** (`phase0_tools/`): build taxonomy, cluster near-duplicate
   images into pseudo-specimens, and produce specimen-aware ID/OOD splits (40/10
   and 30/20).
2. **ID classification** (`engine.py`, `models.py`): train the backbones with a
   CE or IsoMax+ head.
3. **FRIO**: compute the high-frequency ratio and calibration with `freq.py`
   (FASC, test time); the frequency-consistency and frequency-outlier-exposure
   regularizers (FCR/FOE) combine `freq.py`, `eool.py`, and `isomax.py` during
   training.
4. **Evaluation under corruption** (`ood_eval.py`, `corruption.py`): report AUROC,
   FPR@95, and the percentage of ID images flagged as OOD across the corruption
   suite, on both splits, over three seeds.

The end-to-end run is orchestrated by a Google Colab notebook whose cells
(`F00`–`F32`) are indexed in `NOTEBOOK_MAP.md`.

## Notes on the method

- **FASC** is the general, backbone-agnostic, test-time contribution.
- **FCR/FOE** are training-time regularizers; in our experiments they help on
  EfficientNet-B1 but do not transfer to other backbones, so they are presented
  as a backbone-specific enhancement.
- The diffusion-based outlier synthesis (DREAM-OOD / NPOS style) is included for
  completeness; it produced **no measurable benefit** in this domain and is
  reported as a negative result in the paper.

## Citation

```bibtex
@article{macong2026woodood,
  title  = {Distance-Based OOD Detection is Brittle to Frequency Degradation:
            A Fine-Grained Wood Benchmark and a Frequency-Robust Remedy},
  author = {Ma-Cong, Thanh and Bui-Quoc, Bao and Nguyen-Gia, Khanh and
            Tran-Anh, Dat and Nguyen-Trong, Khanh},
  year   = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`).
