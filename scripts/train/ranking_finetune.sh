#!/bin/bash
#SBATCH --job-name=ranking_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=a100,h100
#SBATCH --gres=gpu:4
#SBATCH --account=[insert your SLURM account]
#SBATCH --time=72:00:00
#SBATCH --qos=normal
#SBATCH --error=slogs/ranking_finetune_%j.err
#SBATCH --output=slogs/ranking_finetune_%j.out

# Repo root: defaults to two levels up from this script; override REPO_ROOT to relocate.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slogs

#### Environment and monitoring
# Set WANDB_API_KEY in your shell before submitting (do not hard-code secrets).
export WANDB_MODE="${WANDB_MODE:-online}"
export HYDRA_FULL_ERROR=1
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)

# background GPU monitor
LOG_GPU_FILE="slogs/ranking_finetune_${SLURM_JOB_ID}_gpu_usage.log"
while true; do
    nvidia-smi >> "$LOG_GPU_FILE"
    sleep 15
done &
MONITOR_PID=$!
trap "kill $MONITOR_PID" EXIT

#### Paths and config overrides (relative to REPO_ROOT; override any via env)
# Baseline DFMDock checkpoint to fine-tune.
CKPT_PATH="${CKPT_PATH:-checkpoints/dfmdock_baseline.ckpt}"
# Train/val manifests: JSON lists of
#   {"id", "training_pose": <gt .pt>, "ranking_poses": [[dockq, <decoy .pt>], ...]}
# assembled from the data_gen output (see scripts/README.md).
RANKING_TRAIN_SET="${RANKING_TRAIN_SET:-data/ranking_finetune/ranking_train_set.json}"
RANKING_VAL_SET="${RANKING_VAL_SET:-data/ranking_finetune/ranking_val_set.json}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/ranking_finetune_ckpts/${SLURM_JOB_ID}}"
LR=1e-4
NUM_GPUS=4
EPOCHS=50
CROP_SIZE=1000
VAL_CHECK_INTERVAL=0.5
WEIGHT_DECAY=0.0
RANKING_LOSS_WEIGHT=0.5
SCALING_COEFF_FOR_RANKING_LOSS=10.0
NUM_RANKING_POSES=10
BUCKETS=10
NUM_WORKERS=16
JOB_NAME="$EPOCHS epochs, lr=$LR, wd=$WEIGHT_DECAY, crop_size=$CROP_SIZE, ranking_loss_weight=$RANKING_LOSS_WEIGHT"

#### Print job info
echo "Starting ranking-finetune job"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $JOB_NAME"
echo "Checkpoint: $CKPT_PATH"
echo "Train set: $RANKING_TRAIN_SET"
echo "Output dir: $OUTPUT_DIR"
echo "----------------------------------------"

#### Launch distributed training with Hydra overrides
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT src/Ranking_Net.py \
    --config-path ../configs \
    --config-name ranking_finetune \
    data.ranking_train_set="$RANKING_TRAIN_SET" \
    data.ranking_val_set="$RANKING_VAL_SET" \
    ranking_conf.ckpt_path="$CKPT_PATH" \
    ranking_conf.output_dir="$OUTPUT_DIR" \
    ranking_conf.max_epochs="$EPOCHS" \
    ranking_conf.crop_size="$CROP_SIZE" \
    ranking_conf.lr="$LR" \
    ranking_conf.weight_decay="$WEIGHT_DECAY" \
    ranking_conf.val_check_interval="$VAL_CHECK_INTERVAL" \
    ranking_conf.ranking_loss_weight="$RANKING_LOSS_WEIGHT" \
    +ranking_conf.scaling_coeff_for_ranking_loss="$SCALING_COEFF_FOR_RANKING_LOSS" \
    ranking_conf.num_ranking_poses="$NUM_RANKING_POSES" \
    ranking_conf.buckets="$BUCKETS" \
    ranking_conf.num_workers="$NUM_WORKERS" \
    trainer.devices=$NUM_GPUS \
    wandb.project=DFMDock2 \
    wandb.run_name="ranking_finetune_run_${SLURM_JOB_ID}" \
    +trainer.strategy=ddp_find_unused_parameters_true \

