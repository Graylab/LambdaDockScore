#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=l40s,h100,a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --time=36:00:00
#SBATCH --account=jgray21
#SBATCH --job-name=preprocess_dfmdock
#SBATCH --output=slogs/preprocess_dfmdock_%j.out

set -e # Exit immediately if a command exits with a non-zero status.

# Change to the working directory
cd /scratch/jgray21/rzhu41/eudockscore_versus_dfmdock

# --- Environment Setup ---
# Set TORCH_HOME to a directory on the scratch space to avoid filling up the home directory
export TORCH_HOME="/scratch/jgray21/rzhu41/.cache/torch"
mkdir -p $TORCH_HOME

# Ensure DFMDock sources are discoverable
export PYTHONPATH="/scratch/jgray21/rzhu41/DFMDock/src:${PYTHONPATH}"

# --- Configuration ---
EXTRACTED_DIR="capri_decoys_extracted"
METADATA_FILE="${EXTRACTED_DIR}/decoy_metadata.json"
PT_DIR="dfmdock_pt_files"

# --- Step 2: Preprocessing for DFMDock (PDB to PT) ---
echo "### Step 2: Preprocessing for DFMDock (PDB to PT) ###"
python3 02_preprocess_dfmdock.py \
    --extracted_dir "$EXTRACTED_DIR" \
    --metadata_file "$METADATA_FILE" \
    --output_dir "$PT_DIR"
echo "### Step 2 complete ###"
