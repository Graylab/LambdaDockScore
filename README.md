# LambdaDockScore

Code, checkpoints and results for **"Improving scoring functions for protein-protein
docking with LambdaLoss"** (2025 Machine Learning for Structural Biology Workshop).

LambdaDockScore improves protein-protein pose ranking by fine-tuning the energy
prediction head of [DFMDock](https://doi.org/10.1101/2024.09.27.615401) with a
**LambdaLoss** ranking objective adapted from Learning-to-Rank, using an augmented
dataset of 2.9M decoy poses spanning the full DockQ quality range.

---

## What LambdaDockScore *is*

LambdaDockScore is a **ranking head**, not a standalone docking model. At inference
time it is paired with the unmodified (vanilla) DFMDock sampler:

| Role | Checkpoint |
|---|---|
| Pose sampler **and** baseline energy head | `checkpoints/dfmdock_baseline.ckpt` |
| LambdaDockScore ranking / energy head | `checkpoints/lambdadockscore.ckpt` |

`lambdadockscore.ckpt` is epoch 36 of fine-tuning run `308593` — the best ranking
checkpoint. On benchmarks with pre-existing decoys (CAPRI score set) only the
ranking head is used; on benchmarks without them (DB5.5) the baseline model samples
poses and both heads score the *same* poses, so differences isolate ranking quality.

---

## Repository layout

```
src/                 model + training + inference code
  Ranking_Net.py       LambdaLoss objective and fine-tuning entry point
  DockDPO_data_gen.py  decoy generation (perturbation + DFMDock sampling)
  models/ utils/ datasets/
configs/             Hydra configs (ranking_finetune.yaml drives fine-tuning)
checkpoints/         the two checkpoints above (Git LFS)
scripts/
  train/             SLURM: ranking fine-tuning
  data_gen/          SLURM: decoy dataset generation
  eval/              SLURM + Python: CAPRI and DB5.5 scoring pipelines
analysis/            metrics library + figure regeneration + new analyses
results/             archived per-pose scores backing every figure
figures/             all published figures (bar charts + top-k panels)
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

```bash
cd analysis
python make_paper_figures.py --outdir ../figures
python topk_curves.py --dataset capri --panel
python topk_curves.py --dataset db55 --panel
```

`analysis/lds_metrics.py` is the single source of truth for top-k success rate,
oracle rate and bootstrap confidence intervals; every figure and follow-up
experiment calls it, so definitions cannot drift between analyses.

### Result files

| Path | Contents |
|---|---|
| `results/capri_score_set/*.csv` | 76,078 (target, decoy) pairs over 69 targets, scored by all three models on an identical decoy set |
| `results/db55/{baseline,lambdadockscore}/` | 253 complexes x 120 DFMDock-sampled poses, scored by each head |
| `results/metadata/db55_all_metadata.csv` | DB5.5 category (`AA` = antibody-antigen) and interface size (`delta_asa`) |

---

## Two estimators — use the right one

The paper reports **two different statistics**, and using the wrong one makes
correct data look wrong:

| Figure | Error bars | Estimator | Function |
|---|---|---|---|
| Figure 1 (CAPRI) | no | point estimate | `M.success_at_k` |
| Figures 2, 3, S2 (DB5.5) | yes | pose-level bootstrap, 10,000 reps | `M.pose_bootstrap` |

The DB5.5 bootstrap resamples each complex's poses **with replacement**, re-ranks,
and recomputes success. Because a resample holds only ~63% of the distinct poses,
its mean sits *below* the point estimate when good poses are rare — the DB5.5
oracle is 36.0% as a point estimate but **31.1%** under the bootstrap, which is
what the paper reports.

---

## Training

Fine-tuning ran on 4xA100 for 50 epochs (~35 h); epoch 36 was selected.

```bash
sbatch scripts/train/ranking_finetune.sh
```

Key hyperparameters (`configs/ranking_finetune.yaml`, overridable in the sbatch
script): `lr=1e-4`, `ranking_loss_weight=0.5`, `num_ranking_poses=10`,
`buckets=10`, `crop_size=1000`, `weight_decay=0.0`.

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