#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=l40s,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=72:00:00
#SBATCH --account=jgray21

# Parse part number argument (1-6)
if [ $# -ne 1 ]; then
    echo "Usage: $0 <part_number (1-6)>"
    exit 1
fi

PART_NUM="$1"
if ! [[ "$PART_NUM" =~ ^[1-6]$ ]]; then
    echo "Error: part_number must be an integer from 1 to 6."
    exit 1
fi

# Set output/error file names with part number
#SBATCH --output=slogs/score_dfmdock_part_${PART_NUM}_%j.out
#SBATCH --error=slogs/score_dfmdock_part_${PART_NUM}_%j.err

mkdir -p slogs

# Activate conda environment if needed
conda init
conda activate newEnv

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/scratch/jgray21/rzhu41/eudockscore/src

TARGET_LIST="/scratch/jgray21/rzhu41/eudockscore_versus_dfmdock/filtered_capri_targets_list/segmented_into_six/part_${PART_NUM}.txt"
UPLOADER_DIR="uploaders/database"
EXTRACTED_DIR="capri_decoys_extracted"
METADATA_FILE="${EXTRACTED_DIR}/decoy_metadata.json"
PT_DIR="dfmdock_pt_files"
FINETUNED_CHECKPOINT="/scratch/jgray21/rzhu41/DFMDock/checkpoints/test_ckpts/dips_hetero/308593_epoch_36-step_99197.ckpt"
BASELINE_CHECKPOINT="/scratch/jgray21/rzhu41/DFMDock/checkpoints/test_ckpts/dips_hetero/model_0.ckpt"
DFMDOCK_SCORES="/scratch/jgray21/rzhu41/eudockscore_versus_dfmdock/capri_score_set_results/dfmdock_scores_part_${PART_NUM}.csv"
EUDOCKSCORE_LMDB_DIR="eudockscore_lmdb"
EUDOCKSCORE_SCORES_DIR="eudockscore_scores"

python3 03_score_dfmdock.py \
    --pt_dir "$PT_DIR" \
    --baseline_checkpoint "$BASELINE_CHECKPOINT" \
    --finetuned_checkpoint "$FINETUNED_CHECKPOINT" \
    --output_csv "$DFMDOCK_SCORES" \
    --target_list "$TARGET_LIST"