#!/usr/bin/env python3
"""
Extract individual decoy models from multi-model CAPRI PDB files.

This script:
1. Reads the target list
2. For each target, parses the CSV file to get model metadata (including DockQ scores)
3. Splits the multi-model PDB file into individual PDB files
4. Organizes files in the same structure as the previous pipeline:
    target_name/pdb/target_name_model_X.pdb

Usage:
     python3 01_extract_decoys.py \
          --target_list filtered_capri_targets_list/nonoverlapping_with_dips_complexes.txt \
          --uploader_dir uploaders/database \
          --output_dir capri_decoys_extracted \
          --num_workers 4
"""

import argparse
import os
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
from tqdm import tqdm
import json
import multiprocessing
from functools import partial
import gc
import re


def extract_decoys_from_multimodel_pdb(target_id, uploader_dir, output_dir):
     """
     Extract individual decoy models from a multi-model PDB file, matching by 'identification' column.

     Args:
          target_id: Target ID (e.g., 'T073.2')
          uploader_dir: Directory containing U-*.pdb and U-*.csv files
          output_dir: Output directory for extracted decoys

     Returns:
          Dictionary mapping model filenames to their metadata (including DockQ)
     """

     pdb_file = os.path.join(uploader_dir, f"U-{target_id}.pdb")
     csv_file = os.path.join(uploader_dir, f"U-{target_id}.csv")

     if not os.path.exists(pdb_file):
          return {}

     if not os.path.exists(csv_file):
          return {}

     df = pd.read_csv(csv_file)

     target_output_dir = os.path.join(output_dir, target_id)
     pdb_output_dir = os.path.join(target_output_dir, "pdb")
     os.makedirs(pdb_output_dir, exist_ok=True)

     # Parse the multi-model PDB file manually to map identification to model content
     # We'll build a dict: identification -> pdb_lines
     id_to_pdb_lines = {}
     current_lines = []
     current_identification = None
     in_model = False

     with open(pdb_file, "r") as f:
          for line in f:
               if line.startswith("MODEL"):
                    in_model = True
                    current_lines = [line]
                    current_identification = None
               elif line.startswith("REMARK   4 FROM "):
                    # Example: REMARK   4 FROM T37_U02.M02   SEQID AC  77.5  ID abe9f570cb60991473ea2c1966e306e5
                    m = re.match(r"REMARK\s+4\s+FROM\s+(\S+)", line)
                    if m:
                         current_identification = m.group(1)
                    current_lines.append(line)
               elif line.startswith("ENDMDL"):
                    current_lines.append(line)
                    if current_identification:
                         id_to_pdb_lines[current_identification] = list(current_lines)
                    in_model = False
                    current_lines = []
                    current_identification = None
               elif in_model:
                    current_lines.append(line)
               # lines outside MODEL/ENDMDL are ignored

     model_metadata = {}

     for idx, row in df.iterrows():
          identification = str(row['identification'])
          dockq = float(row['dockq']) if pd.notna(row['dockq']) else None

          output_filename = f"{target_id}_{identification}.pdb"
          output_path = os.path.join(pdb_output_dir, output_filename)

          pdb_lines = id_to_pdb_lines.get(identification)
          if not pdb_lines:
               # print(f"    Warning: Identification {identification} not found in PDB file")
               continue

          with open(output_path, "w") as out_f:
               out_f.writelines(pdb_lines)

          model_metadata[output_filename] = {
               'target_id': target_id,
               'identification': identification,
               'model_num': int(row['model']) if pd.notna(row['model']) else None,
               'dockq': dockq,
               'pdb_path': output_path,
               'fnat': float(row['fnat']) if pd.notna(row['fnat']) else None,
               'irms': float(row['irms']) if pd.notna(row['irms']) else None,
               'lrms': float(row['lrms']) if pd.notna(row['lrms']) else None,
               'classification': row['classification']
          }

     gc.collect()
     return model_metadata


def main():
     parser = argparse.ArgumentParser(description='Extract CAPRI decoys from multi-model PDB files')
     parser.add_argument('--target_list', type=str, required=True,
                              help='Path to file containing target IDs (one per line)')
     parser.add_argument('--uploader_dir', type=str, required=True,
                              help='Directory containing U-*.pdb and U-*.csv files')
     parser.add_argument('--output_dir', type=str, required=True,
                              help='Output directory for extracted decoys')
     parser.add_argument('--metadata_output', type=str, default=None,
                              help='Path to save metadata JSON file')
     parser.add_argument('--num_workers', type=int, default=None,
                                help='Number of CPU cores to use for parallel processing. Defaults to SLURM_CPUS_PER_TASK or os.cpu_count()')
     
     args = parser.parse_args()
     
     # Read target list
     with open(args.target_list, 'r') as f:
          target_ids = [line.strip() for line in f if line.strip()]
     
     # Determine the number of workers
     if args.num_workers:
          num_workers = args.num_workers
     else:
          slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
          num_workers = int(slurm_cpus) if slurm_cpus else os.cpu_count()
     
     # Don't use more workers than targets
     num_workers = min(num_workers, len(target_ids))
     if num_workers < 1:
          num_workers = 1

     print(f"Processing {len(target_ids)} targets using {num_workers} workers...")
     
     # Process each target in parallel
     all_metadata = {}
     
     # Create a partial function with fixed arguments
     worker_func = partial(extract_decoys_from_multimodel_pdb, 
                                  uploader_dir=args.uploader_dir, 
                                  output_dir=args.output_dir)

     with multiprocessing.Pool(processes=num_workers) as pool:
          # Use tqdm to show progress bar
          results = list(tqdm(pool.imap_unordered(worker_func, target_ids, chunksize=1), total=len(target_ids), desc="Extracting decoys"))

     # Combine results from all processes
     for metadata in results:
          all_metadata.update(metadata)
     
     print(f"\nTotal models extracted: {len(all_metadata)}")
     
     # Save metadata to JSON
     if args.metadata_output:
          metadata_path = args.metadata_output
     else:
          metadata_path = os.path.join(args.output_dir, 'decoy_metadata.json')
     
     with open(metadata_path, 'w') as f:
          json.dump(all_metadata, f, indent=2)
     
     print(f"Metadata saved to {metadata_path}")
     
     # Print summary statistics
     print("\n=== Summary ===")
     print(f"Total targets processed: {len(target_ids)}")
     print(f"Total decoys extracted: {len(all_metadata)}")
     
     # DockQ distribution
     dockq_values = [m['dockq'] for m in all_metadata.values() if m['dockq'] is not None]
     if dockq_values:
          print(f"DockQ range: {min(dockq_values):.4f} - {max(dockq_values):.4f}")
          print(f"Average DockQ: {sum(dockq_values)/len(dockq_values):.4f}")
     
     # Classification distribution
     classifications = [m['classification'] for m in all_metadata.values()]
     from collections import Counter
     class_counts = Counter(classifications)
     print("\nClassification distribution:")
     for cls, count in class_counts.most_common():
          print(f"  {cls}: {count}")


if __name__ == '__main__':
     main()
