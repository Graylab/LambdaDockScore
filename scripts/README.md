# Scripts

These are the SLURM/Python pipelines used to generate the fine-tuning data, train
LambdaDockScore, and score the two benchmarks. They were written for the Gray Lab
cluster and contain **absolute paths** (e.g. `/scratch/jgray21/rzhu41/...`) and
`#SBATCH` directives (`--account=[insert your SLURM account]`, partitions `a100,h100`, etc.). Edit the
paths, the account, and the partitions for your own environment before running.

The published figures in this repo can be regenerated **without** any of these
scripts — they only read the CSVs in `results/` (see the top-level `README.md`).
These scripts are provided for full provenance and to retrain/re-score from scratch.

## `data_gen/` — build the fine-tuning decoy set

`dpo_data_gen.sh` runs `src/DockDPO_data_gen.py` to generate 270 decoys per
complex (perturbed ground truth + DFMDock samples; see Supplementary Table S1).

| Variable | Meaning |
|---|---|
| `BASE_DIR` | working copy of the DFMDock code + checkpoints |
| `CKPT_DIR/TRAIN_SET/MODEL` | sampler checkpoint used to draw poses (`checkpoints/test_ckpts/dips_hetero/model_0.ckpt`) |
| `POSE_DATASET` | source complexes to build decoys from (`db5_all`, or the DIPS-hetero set for training data) |
| `POSE_SET_SUBSET_INDICES` | `[start,end]` slice of the complex list, so generation can be sharded across jobs |
| `OUTPUT_DIR` | where the generated `.pt`/`.json` decoys are written |

The source complex set is DIPS-hetero (Townshend et al. 2019; Morehead et al. 2023);
download it via the DIPS-Plus release and point `POSE_DATASET` at it. The full
generated set (~2.9M poses) is too large to distribute.

## `train/` — fine-tune the ranking head

`ranking_finetune.sh` runs `src/Ranking_Net.py` (Hydra config
`configs/ranking_finetune.yaml`) with the LambdaLoss objective.

| Variable | Meaning / how to produce it |
|---|---|
| `CKPT_PATH` | baseline DFMDock checkpoint to fine-tune (`checkpoints/dfmdock_baseline.ckpt`) |
| `RANKING_TRAIN_SET` / `RANKING_VAL_SET` | JSON manifests (see format below) |
| `OUTPUT_DIR` | where fine-tuned checkpoints are written (epoch 36 = `lambdadockscore.ckpt`) |

**Building the train/val manifests.** `data_gen/` writes one
`individual_poses.json` per complex (pose `.pt` paths + their DockQ). `Ranking_Net.py`
instead expects a single JSON *list*, one entry per complex, in this schema:

```json
[
  {"id": "1abc",
   "training_pose": "path/to/ground_truth.pt",
   "ranking_poses": [[0.83, "path/to/decoy_0.pt"], [0.12, "path/to/decoy_1.pt"]]}
]
```

Concatenate the per-complex `data_gen` outputs into that shape and split the list
of complexes into train and validation partitions to produce the two JSON files.

Set `WANDB_API_KEY` in your shell before submitting (the script no longer hard-codes it).
Key hyperparameters are the `LR`, `RANKING_LOSS_WEIGHT`, `NUM_RANKING_POSES`,
`BUCKETS`, and `CROP_SIZE` variables near the top of the script.

## `eval/` — score the benchmarks

**CAPRI score set** (pre-existing decoys, numbered 01–06):

| Step | Script | What it does |
|---|---|---|
| 01 | `01_extract_decoys.py` / `.sh` | pull decoy PDBs out of the CAPRI score-set uploader files (`U-<target>.pdb/.csv`); writes `decoy_metadata.json` |
| 02 | `02_preprocess_dfmdock.py` / `.sh` | turn decoy PDBs into DFMDock graph `.pt` files |
| 03 | `03_score_dfmdock.py` / `.sh` | score every `.pt` with the baseline **and** fine-tuned heads → `*_baseline.csv`, `*_finetuned.csv` |
| 04 | `04_preprocess_eudockscore.sh` | build EuDockScore LMDBs from the same decoys |
| 05 | `05_score_eudockscore.sh` | score the decoys with EuDockScore |
| 06 | `06_calculate_metrics.py` | merge scores with ground-truth DockQ, compute top-k / oracle |

Consolidated outputs of this pipeline are the CSVs already committed under
`results/capri_score_set/`.

**DB5.5** (no pre-existing decoys): `db55_sample_and_score.sh` samples poses with
baseline DFMDock and scores them with both heads in one pass, producing the CSVs
under `results/db55/`.

Key hardcoded paths in the eval scripts:

| Variable | Meaning |
|---|---|
| `BASELINE_CHECKPOINT` | `model_0.ckpt` (= `checkpoints/dfmdock_baseline.ckpt`) |
| `FINETUNED_CHECKPOINT` / `RANKING_MODEL` | epoch-36 fine-tune `308593_epoch_36-step_99197.ckpt` (= `checkpoints/lambdadockscore.ckpt`) |
| `TARGET_LIST` | text file of CAPRI target IDs kept after <30% identity filtering; `part_N.txt` are six shards for parallel scoring |
| `UPLOADER_DIR` | raw CAPRI score-set uploader files (`U-<target>.pdb/.csv`) from the CAPRI website |
| `EXTRACTED_DIR` / `PT_DIR` / `*_LMDB_DIR` | intermediate decoy PDBs, DFMDock `.pt` graphs, and EuDockScore LMDBs produced by steps 01/02/04 |

EuDockScore itself (the `eudockscore/src` on `PYTHONPATH` and the
`run_eudockscore_data.py` runner) is the external package from McFee et al. 2024;
install it separately and repoint those paths.
