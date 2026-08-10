#!/bin/bash
#SBATCH --job-name=ranking_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=a100,h100
#SBATCH --gres=gpu:4
#SBATCH --account=jgray21
#SBATCH --time=72:00:00
#SBATCH --qos=normal
#SBATCH --error=/scratch/jgray21/rzhu41/DFMDock/slogs/slogs_ranking_finetune/%j.err
#SBATCH --output=/scratch/jgray21/rzhu41/DFMDock/slogs/slogs_ranking_finetune/%j.out

#### Environment and monitoring
export WANDB_API_KEY="2f73cace6c721897dade5cffd271e3d1c1a95faa"
export WANDB_MODE="online"
export HYDRA_FULL_ERROR=1
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)

# background GPU monitor
LOG_GPU_FILE="/scratch/jgray21/rzhu41/DFMDock/slogs/slogs_ranking_finetune/${SLURM_JOB_ID}_gpu_usage.log"
while true; do
    nvidia-smi >> "$LOG_GPU_FILE"
    sleep 15
done &
MONITOR_PID=$!
trap "kill $MONITOR_PID" EXIT

#### Paths and config overrides
CKPT_PATH="/scratch/jgray21/rzhu41/DFMDock/checkpoints/test_ckpts/dips_hetero/model_0.ckpt"
RANKING_TRAIN_SET="/scratch/jgray21/rzhu41/DFMDock2/data/ranking_finetune/ranking_train_set_july17_removed_zero_bytes_removed_corrupted.json"
RANKING_VAL_SET="/scratch/jgray21/rzhu41/DFMDock2/data/ranking_finetune/ranking_val_set_july17_removed_zero_bytes_removed_corrupted.json"
OUTPUT_DIR="/scratch/jgray21/rzhu41/DFMDock/checkpoints/ranking_finetune_ckpts/${SLURM_JOB_ID}"
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
echo "Using checkpoint: $CKPT_DIR"
echo "Data list: $DATA_LIST"
echo "Output dir: $OUTPUT_DIR"
echo "----------------------------------------"

#### Launch distributed training with Hydra overrides
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT /scratch/jgray21/rzhu41/DFMDock/src/Ranking_Net.py \
    --config-path /scratch/jgray21/rzhu41/DFMDock/configs \
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

