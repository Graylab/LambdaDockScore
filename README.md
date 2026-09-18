# LambdaDockScore

Code, checkpoints and results for **"Improving scoring functions for protein-protein
docking with LambdaLoss"** (2025 Machine Learning for Structural Biology Workshop).
Code, checkpoints, and results for **"Improving scoring functions for protein-protein docking with LambdaLoss"** (2025 Machine Learning for Structural Biology Workshop).

LambdaDockScore improves protein-protein pose ranking by fine-tuning the energy
prediction head of [DFMDock](https://doi.org/10.1101/2024.09.27.615401) with a
**LambdaLoss** ranking objective adapted from Learning-to-Rank, using an augmented
dataset of 2.9M decoy poses spanning the full DockQ quality range.
LambdaDockScore improves protein-protein pose ranking by fine-tuning the energy prediction head of [DFMDock](https://doi.org/10.1101/2024.09.27.615401) with a **LambdaLoss** ranking objective adapted from Learning-to-Rank, trained on an augmented dataset of 2.9M decoy poses spanning the full DockQ quality range.

---

## What LambdaDockScore *is*
## Overview

LambdaDockScore is a **ranking head**, not a standalone docking model. At inference
time it is paired with the unmodified (vanilla) DFMDock sampler:
LambdaDockScore functions as a ranking head rather than a standalone docking model. At inference time, it is paired with the unmodified DFMDock sampler:

| Role | Checkpoint |
|---|---|
| Pose sampler **and** baseline energy head | `checkpoints/dfmdock_baseline.ckpt` |
| Pose sampler and baseline energy head | `checkpoints/dfmdock_baseline.ckpt` |
| LambdaDockScore ranking / energy head | `checkpoints/lambdadockscore.ckpt` |

`lambdadockscore.ckpt` is epoch 36 of fine-tuning run `308593` — the best ranking
checkpoint. On benchmarks with pre-existing decoys (CAPRI score set) only the
ranking head is used; on benchmarks without them (DB5.5) the baseline model samples
poses and both heads score the *same* poses, so differences isolate ranking quality.
The checkpoint `checkpoints/lambdadockscore.ckpt` corresponds to epoch 36 of fine-tuning run `308593`, selected based on validation ranking performance. On benchmarks with pre-existing decoys (the CAPRI score set), only the ranking head is evaluated. On benchmarks without pre-existing decoys (DB5.5), the baseline model samples poses and both scoring heads evaluate identical pose sets, isolating scoring and ranking performance.

---

## Repository layout

```
src/                 model + training + inference code
  Ranking_Net.py       LambdaLoss objective and fine-tuning entry point
  DockDPO_data_gen.py  decoy generation (perturbation + DFMDock sampling)
  models/ utils/ datasets/
configs/             Hydra configs (ranking_finetune.yaml drives fine-tuning)
checkpoints/         the two checkpoints above (Git LFS)
checkpoints/         model checkpoints (Git LFS)
scripts/
  train/             SLURM: ranking fine-tuning
  data_gen/          SLURM: decoy dataset generation
  eval/              SLURM + Python: CAPRI and DB5.5 scoring pipelines
analysis/            metrics library + figure regeneration + new analyses
analysis/            metrics library + figure regeneration
results/             archived per-pose scores backing every figure
figures/             all published figures (bar charts + top-k panels)
figures/             published figures (bar charts and top-k panels)
```

## Installation

```bash
conda env create -f environment.yml
conda activate lambdadockscore
```

Checkpoints are stored with Git LFS:

```bash
git lfs install && git lfs pull
```

---

## Reproducing the analysis

All published numbers derive from the per-pose CSVs in `results/`, so the figures
regenerate without a GPU:
All published numbers derive from the per-pose CSVs in `results/`, allowing figures to be regenerated without a GPU:

```bash
cd analysis
python make_paper_figures.py --outdir ../figures
python topk_curves.py --dataset capri --panel
python topk_curves.py --dataset db55 --panel
```

`analysis/lds_metrics.py` is the single source of truth for top-k success rate,
oracle rate and bootstrap confidence intervals; every figure and follow-up
experiment calls it, so definitions cannot drift between analyses.
`analysis/lds_metrics.py` provides the canonical implementations for top-k success rates, oracle rates, and bootstrap confidence intervals across all figures.

### Result files

| Path | Contents |
|---|---|
| `results/capri_score_set/*.csv` | 76,078 (target, decoy) pairs over 69 targets, scored by all three models on an identical decoy set |
| `results/capri_score_set/*.csv` | 76,078 (target, decoy) pairs across 69 targets, scored by all three models on an identical decoy set |
| `results/db55/{baseline,lambdadockscore}/` | 253 complexes x 120 DFMDock-sampled poses, scored by each head |
| `results/metadata/db55_all_metadata.csv` | DB5.5 category (`AA` = antibody-antigen) and interface size (`delta_asa`) |

**Note on CAPRI target filtering:** While 76 CAPRI targets pass sequence-identity filtering against the training set (<30% sequence identity to DIPS-hetero), 7 targets are three-chain trimer complexes (T025.1, T037.1, T037.2, T050.1, T081.1, T122.1, T165.1). Because both DFMDock and EuDockScore assume two-body complexes, these three-chain targets encountered preprocessing limitations (such as empty receptor graphs in DFMDock) or runtime exceptions in EuDockScore. Requiring valid scores across all three models (a 3-way intersection) yielded the 69 two-body dimer targets evaluated in the paper.

---

## Two estimators — use the right one
## Evaluation estimators

The paper reports **two different statistics**, and using the wrong one makes
correct data look wrong:
The evaluation reports two different statistics depending on the benchmark:

| Figure | Error bars | Estimator | Function |
|---|---|---|---|
| Figure 1 (CAPRI) | no | point estimate | `M.success_at_k` |
| Figures 2, 3, S2 (DB5.5) | yes | pose-level bootstrap, 10,000 reps | `M.pose_bootstrap` |
| Figure 1 (CAPRI) | None | Point estimate | `M.success_at_k` |
| Figures 2, 3, S2 (DB5.5) | 95% CI | Pose-level bootstrap (10,000 replicates) | `M.pose_bootstrap` |

The DB5.5 bootstrap resamples each complex's poses **with replacement**, re-ranks,
and recomputes success. Because a resample holds only ~63% of the distinct poses,
its mean sits *below* the point estimate when good poses are rare — the DB5.5
oracle is 36.0% as a point estimate but **31.1%** under the bootstrap, which is
what the paper reports.
The DB5.5 evaluation computes a pose-level bootstrap by resampling each complex's poses with replacement over 10,000 replicates, re-ranking them, and calculating the mean success rate.

---

## Training

Fine-tuning ran on 4xA100 for 50 epochs (~35 h); epoch 36 was selected.

```bash
sbatch scripts/train/ranking_finetune.sh
```

Key hyperparameters (`configs/ranking_finetune.yaml`, overridable in the sbatch
script): `lr=1e-4`, `ranking_loss_weight=0.5`, `num_ranking_poses=10`,
`buckets=10`, `crop_size=1000`, `weight_decay=0.0`.
Key hyperparameters (`configs/ranking_finetune.yaml`, overridable in the sbatch script): `lr=1e-4`, `ranking_loss_weight=0.5`, `num_ranking_poses=10`, `buckets=10`, `crop_size=1000`, `weight_decay=0.0`.

### Decoy generation

270 decoys per complex, matching Supplementary Table S1:

| Decoy set | Translation sigma (A) | Rotation sigma (deg) | # poses |
|---|---|---|---|
| Perturbed GT - small | 0.1 | 1.0 | 20 |
| Perturbed GT - medium | 1.0 | 7.0 | 100 |
| Perturbed GT - large | 2.5 | 20.0 | 100 |
| DFMDock samples | - | - | 50 |

```bash
sbatch scripts/data_gen/dpo_data_gen.sh
```

The generated decoy dataset (~2.9M poses) is far too large to distribute; the
scripts regenerate it from DIPS-hetero.
Because the generated decoy dataset (~2.9M poses) requires over 2 TB of storage, the decoy generation scripts are provided to regenerate poses from DIPS-hetero.

See `scripts/README.md` for the full data-generation, training, and scoring
pipelines, including every configurable path and how to build the train/val
manifests. The scripts use repo-relative paths (override `REPO_ROOT`, `DATA_ROOT`,
`DFMDOCK_SRC`, `EUDOCKSCORE_SRC`, `WORKDIR`, etc.) and SLURM directives you will
need to adapt to your cluster.
See `scripts/README.md` for the full data-generation, training, and scoring pipelines, including configurable paths and dataset manifest formatting.

---

## Citation

```bibtex
@inproceedings{zhu2025lambdadockscore,
  title     = {Improving scoring functions for protein-protein docking with LambdaLoss},
  author    = {Zhu, Richard and Xu, Darren and Chu, Lee-Shin and Gray, Jeffrey J.},
  booktitle = {Machine Learning for Structural Biology Workshop},
  year      = {2025}
}
```

## License

MIT — see `LICENSE`.
This project is licensed under the MIT License (see `LICENSE`).