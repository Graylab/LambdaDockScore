#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=l40s,h100,a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --time=36:00:00
#SBATCH --account=jgray21
#SBATCH --job-name=extract_decoys
#SBATCH --output=slogs/extract_decoys_%j.out

set -e # Exit immediately if a command exits with a non-zero status.

# Change to the working directory
cd /scratch/jgray21/rzhu41/eudockscore_versus_dfmdock

# --- Configuration ---
TARGET_LIST="filtered_capri_targets_list/nonoverlapping_with_dips_complexes.txt"
UPLOADER_DIR="uploaders/database"
EXTRACTED_DIR="capri_decoys_extracted"
METADATA_FILE="${EXTRACTED_DIR}/decoy_metadata.json"

# --- Step 1: Extracting decoys ---
echo "### Step 1: Extracting decoys ###"
python3 01_extract_decoys.py \
    --target_list "$TARGET_LIST" \
    --uploader_dir "$UPLOADER_DIR" \
    --output_dir "$EXTRACTED_DIR" \
    --metadata_output "$METADATA_FILE" \
    --num_workers "${SLURM_CPUS_PER_TASK:-1}"
echo "### Step 1 complete ###"
