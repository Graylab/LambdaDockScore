#!/bin/bash

cd "${WORKDIR:-.}"

EXTRACTED_DIR="${EXTRACTED_DIR:-capri_decoys_extracted}"
TARGET_LIST="${TARGET_LIST:-filtered_capri_targets_list/nonoverlapping_with_dips_complexes.txt}"
LMDB_DIR="${LMDB_DIR:-eudockscore_lmdb}"

mkdir -p "$LMDB_DIR"

process_target() {
    target_id="$1"
    pdb_dir="$EXTRACTED_DIR/$target_id/pdb"
    lmdb_output="$LMDB_DIR/${target_id}.lmdb"

    if [ -d "$pdb_dir" ]; then
        echo "Processing $target_id..."
        # --- MODIFIED LINE ---
        # Use the general atom3d.datasets command, which is known to work.
        python -m atom3d.datasets "$pdb_dir" "$lmdb_output" --filetype pdb
        echo "Created LMDB: $lmdb_output"
    else
        echo "Warning: PDB directory not found for $target_id"
    fi
}

export EXTRACTED_DIR LMDB_DIR
export -f process_target

N=$(nproc)

cat "$TARGET_LIST" | xargs -I{} -n1 -P"$N" bash -c 'process_target "$0"' {}