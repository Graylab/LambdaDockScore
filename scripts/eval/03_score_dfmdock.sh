#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=l40s,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=72:00:00
#SBATCH --account=[insert your SLURM account]

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
conda activate lambdadockscore

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$PYTHONPATH:${EUDOCKSCORE_SRC:-../eudockscore/src}"

# List of CAPRI target IDs to score; part_N.txt are six shards for parallel jobs.
TARGET_LIST="${TARGET_LIST:-filtered_capri_targets_list/segmented_into_six/part_${PART_NUM}.txt}"
UPLOADER_DIR="uploaders/database"
EXTRACTED_DIR="capri_decoys_extracted"
METADATA_FILE="${EXTRACTED_DIR}/decoy_metadata.json"
PT_DIR="dfmdock_pt_files"
FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:-../../checkpoints/lambdadockscore.ckpt}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-../../checkpoints/dfmdock_baseline.ckpt}"
DFMDOCK_SCORES="${DFMDOCK_SCORES:-capri_score_set_results/dfmdock_scores_part_${PART_NUM}.csv}"
EUDOCKSCORE_LMDB_DIR="eudockscore_lmdb"
EUDOCKSCORE_SCORES_DIR="eudockscore_scores"

python3 03_score_dfmdock.py \
    --pt_dir "$PT_DIR" \
    --baseline_checkpoint "$BASELINE_CHECKPOINT" \
    --finetuned_checkpoint "$FINETUNED_CHECKPOINT" \
    --output_csv "$DFMDOCK_SCORES" \
    --target_list "$TARGET_LIST"