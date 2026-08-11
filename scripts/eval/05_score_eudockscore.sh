#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --partition=l40s,h100,a100
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-gpu=16
#SBATCH --time=72:00:00
#SBATCH --account=[insert your SLURM account]
#SBATCH --output=slogs/score_eudockscore_%j.out
#SBATCH --error=slogs/score_eudockscore_%j.err

mkdir -p slogs

# Activate conda environment if needed
conda activate newEnv

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$PYTHONPATH:${EUDOCKSCORE_SRC:-../eudockscore/src}"

# --- Configuration ---
EXTRACTED_DIR="${EXTRACTED_DIR:-capri_decoys_extracted}"
TARGET_LIST="${TARGET_LIST:-filtered_capri_targets_list/nonoverlapping_with_dips_complexes.txt}"
LMDB_DIR="${LMDB_DIR:-eudockscore_lmdb}"
OUTPUT_DIR="${OUTPUT_DIR:-capri_score_set_eudockscore_results}"
# EuDockScore's own runner script (from the eudockscore package).
EUDOCKSCORE_RUN_SCRIPT="${EUDOCKSCORE_RUN_SCRIPT:-../eudockscore/run_eudockscore_data.py}"

mkdir -p "$OUTPUT_DIR"

# --- Main Logic ---
echo "Starting EuDockScore scoring run..."

# Get the directory of the scoring script
EUDOCKSCORE_RUN_DIR=$(dirname "$EUDOCKSCORE_RUN_SCRIPT")

while IFS= read -r target_id || [[ -n "$target_id" ]]; do
    echo "--------------------------------------------------"
    echo "Processing target: $target_id"
    
    target_dir="$EXTRACTED_DIR/$target_id"
    pdb_dir="$target_dir/pdb"
    
    if [ ! -d "$pdb_dir" ]; then
        echo "Warning: PDB directory not found for $target_id. Skipping."
        continue
    fi

    csv_input_file="$target_dir/${target_id}_pdbs_for_scoring.csv"
    scores_output_file="$OUTPUT_DIR/${target_id}_scores.csv"
    lmdb_path="$LMDB_DIR/${target_id}.lmdb"

    # 1. Generate the required CSV file
    echo "Generating PDB list: $csv_input_file"
    echo "File,Label" > "$csv_input_file"
    find "$pdb_dir" -maxdepth 1 -name "*.pdb" -printf "%f,0\n" >> "$csv_input_file"

    # 2. Run the scoring script from its own directory
    echo "Running EuDockScore for $target_id..."
    (
        cd "$EUDOCKSCORE_RUN_DIR" || exit
        python3 -u "$(basename "$EUDOCKSCORE_RUN_SCRIPT")" \
            --root_dir "$lmdb_path" \
            --csv_name "$csv_input_file" \
            --file_name "$scores_output_file"
    )

    echo "Scoring complete for $target_id. Output: $scores_output_file"

done < "$TARGET_LIST"

echo "--------------------------------------------------"
echo "All EuDockScore scoring tasks finished."

