import os
import argparse
import subprocess
import sys
from multiprocessing import Pool, cpu_count
import functools
from tqdm import tqdm
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import pandas as pd
from collections import Counter

#!/usr/bin/env python3
"""
Pipeline to filter CAPRI Score Set targets by sequence identity against DIPS train/val lists,
and to perform MMseqs2 clustering at 80% coverage.

Steps:
1. For each PDB file in the CAPRI target directory:
    - Check its quality file. If all models are 'incorrect', skip the target.
    - Otherwise, extract all chain sequences and collect quality statistics.
2. Read PDB IDs from DIPS train/val hetero lists and fetch their sequences from .pt files.
3. Combine all CAPRI and DIPS sequences into a single FASTA file.
4. Run MMseqs2 clustering (e.g., `easy-cluster`) at 80% coverage and >=30% identity.
5. Identify CAPRI targets whose chains cluster with any DIPS chain.
6. Remove those targets and report the remaining ones.
7. Save the quality statistics for the included targets to a CSV file.

Usage:
     python3 clean_capri_score_set.py \
          --capri_root /path/to/capri_targets \
          --capri_quality_dir /path/to/quality/files \
          --dips_train /path/to/dips_train_hetero.txt \
          --dips_val /path/to/dips_val_hetero.txt \
          --output /path/to/output/dir/remaining_targets.txt

Dependencies:
     torch, biopython, tqdm, pandas, mmseqs2 (installed in PATH)
"""


def extract_chains_from_pdb(pdb_file, prefix):
    """Extract individual chain sequences from a PDB file using PPBuilder."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
    builder = PPBuilder()
    records = []
    for model in structure:  # Process the first model
        for chain in model:
            peptides = builder.build_peptides(chain)
            if not peptides:
                continue
            # Concatenate all peptide fragments
            full_seq = "".join(str(p.get_sequence()) for p in peptides)
            rec = SeqRecord(Seq(full_seq),
                            id=f"{prefix}_{chain.id}",
                            description="")
            records.append(rec)
        break  # Only process the first model
    return records

def read_dips_list(txt_path):
    """Read PDB IDs (one per line)."""
    ids = []
    with open(txt_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pdb_id = line.split()[0]
            ids.append(pdb_id)
    return ids

def process_dips_id(_id, data_dir):
    """Process a single DIPS ID and return sequence records."""
    records = []
    # Parse ID e.g., '1A2K/1A2K_A_B.pdb' -> '1A2K_1A2K_A_B'
    split_string = _id.split('/')
    parsed_id = split_string[0] + '_' + split_string[1].rsplit('.', 1)[0]
    
    file_path = os.path.join(data_dir, parsed_id + '.pt')
    if not os.path.exists(file_path):
        return records, f"Warning: {file_path} not found; skipping."
    
    try:
        data = torch.load(file_path, weights_only=False)
        receptor_seq = data['receptor'].seq
        ligand_seq = data['ligand'].seq
        
        rec_receptor = SeqRecord(Seq(receptor_seq), id=f"{parsed_id}_receptor", description='')
        records.append(rec_receptor)
        
        rec_ligand = SeqRecord(Seq(ligand_seq), id=f"{parsed_id}_ligand", description='')
        records.append(rec_ligand)
    except Exception as e:
        return records, f"Warning: Failed to load {file_path}: {e}"
    
    return records, None

def main():
    parser = argparse.ArgumentParser(description="Filter CAPRI targets against DIPS dataset.")
    parser.add_argument('--capri_root', required=True, help="Directory containing CAPRI target .pdb files.")
    parser.add_argument('--capri_quality_dir', required=True, help="Directory with CAPRI quality assessment files (U-*.txt).")
    parser.add_argument('--dips_train', required=True, help="Path to DIPS training set list.")
    parser.add_argument('--dips_val', required=True, help="Path to DIPS validation set list.")
    parser.add_argument('--output', required=True, help="Output file for remaining target names. Intermediate files will be saved in its directory.")
    parser.add_argument('--data_dir', default='/scratch/jgray21/rzhu41/DFMDock2/src/data/pt/dips_bb', help="Directory with DIPS .pt files.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract sequences from all CAPRI target PDBs, filtering by quality
    capri_records = []
    capri_stats = []
    capri_pdb_files = [f for f in os.listdir(args.capri_root) if f.lower().endswith('.pdb')]
    print(f"Found {len(capri_pdb_files)} CAPRI PDB files.")
    
    for pdb_filename in tqdm(capri_pdb_files, desc="Processing CAPRI targets"):
        target_id = os.path.splitext(pdb_filename)[0]
        quality_filename = f"U-{target_id}.txt"
        quality_filepath = os.path.join(args.capri_quality_dir, quality_filename)

        if not os.path.exists(quality_filepath):
            print(f"Warning: Quality file not found for {target_id}, skipping: {quality_filepath}")
            continue

        qualities = []
        with open(quality_filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    qualities.append(parts[-1])
        
        if not qualities or all(q == 'incorrect' for q in qualities):
            print(f"Info: Skipping target {target_id} as all models are 'incorrect' or file is empty.")
            continue

        # If target is included, process it
        pdb_path = os.path.join(args.capri_root, pdb_filename)
        try:
            chains = extract_chains_from_pdb(pdb_path, target_id)
            capri_records.extend(chains)
            
            # Gather stats
            counts = Counter(qualities)
            capri_stats.append({
                'target_id': target_id,
                'total_models': len(qualities),
                'incorrect': counts.get('incorrect', 0),
                'acceptable': counts.get('acceptable', 0),
                'medium': counts.get('medium', 0),
                'high': counts.get('high', 0)
            })
        except Exception as e:
            print(f"Warning: Failed to process {pdb_path}: {e}")

    capri_fasta = os.path.join(output_dir, 'capri_sequences.fasta')
    SeqIO.write(capri_records, capri_fasta, 'fasta')
    print(f"Wrote {len(capri_records)} CAPRI chain sequences to {capri_fasta}")

    # 2. Extract DIPS sequences from .pt files
    dips_ids = read_dips_list(args.dips_train) + read_dips_list(args.dips_val)
    dips_records = []
    
    num_processes = min(cpu_count(), 16)
    process_func = functools.partial(process_dips_id, data_dir=args.data_dir)
    
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_func, dips_ids), total=len(dips_ids), desc="Processing DIPS IDs"))
    
    for records, error_msg in results:
        if error_msg:
            print(error_msg, file=sys.stderr)
        dips_records.extend(records)
     
    dips_fasta = os.path.join(output_dir, 'dips_sequences.fasta')
    SeqIO.write(dips_records, dips_fasta, 'fasta')
    print(f"Wrote {len(dips_records)} DIPS chain sequences to {dips_fasta}")

    # 3. Combine FASTA files
    combined_fasta = os.path.join(output_dir, 'combined.fasta')
    with open(combined_fasta, 'wb') as outfile:
        with open(capri_fasta, 'rb') as infile:
            outfile.write(infile.read())
        with open(dips_fasta, 'rb') as infile:
            outfile.write(infile.read())
    print(f"Combined sequences into {combined_fasta}")

    # 4. Run MMseqs2 clustering
    cluster_basename = os.path.join(output_dir, 'clusterRes')
    tmp_dir = os.path.join(output_dir, 'tmp')
    out_clu = f"{cluster_basename}_cluster.tsv"
    
    print("Running MMseqs2 clustering...")
    try:
        subprocess.run(['mmseqs', 'easy-cluster', combined_fasta, cluster_basename, tmp_dir,
                        '--cov-mode', '1', '-c', '0.8', '--min-seq-id', '0.3'], 
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running MMseqs2: {e}", file=sys.stderr)
        if isinstance(e, subprocess.CalledProcessError):
            print(f"MMseqs2 stderr:\n{e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    
    # 5. Identify CAPRI targets with chains overlapping with DIPS chains
    capri_chain_ids = {r.id for r in capri_records}
    dips_chain_ids = {r.id for r in dips_records}
    overlapping_targets = set()

    print(f"Parsing cluster results from {out_clu}...")
    with open(out_clu) as fh:
        for line in fh:
            try:
                rep, member = line.strip().split('\t')
            except ValueError:
                continue

            is_rep_capri = rep in capri_chain_ids
            is_member_capri = member in capri_chain_ids
            is_rep_dips = rep in dips_chain_ids
            is_member_dips = member in dips_chain_ids

            if (is_rep_capri and is_member_dips) or (is_rep_dips and is_member_capri):
                if is_rep_capri:
                    target_id = rep.rsplit('_', 1)[0]
                    overlapping_targets.add(target_id)
                if is_member_capri:
                    target_id = member.rsplit('_', 1)[0]
                    overlapping_targets.add(target_id)

    # 6. Report remaining targets
    all_capri_targets = {r.id.rsplit('_', 1)[0] for r in capri_records}
    remaining_targets = sorted(list(all_capri_targets - overlapping_targets))
    
    print(f"\nTotal CAPRI targets included after quality filter: {len(all_capri_targets)}")
    print(f"Overlapping targets found via MMseqs2: {len(overlapping_targets)}")
    print(f"Remaining non-overlapping targets: {len(remaining_targets)}")

    with open(args.output, 'w') as out:
        for target in remaining_targets:
            out.write(target + '\n')
    print(f"Remaining targets written to {args.output}")

    # 7. Save quality stats for the remaining targets to a CSV file
    if capri_stats:
        stats_df = pd.DataFrame(capri_stats)
        # Filter the DataFrame to only include the remaining targets
        final_stats_df = stats_df[stats_df['target_id'].isin(remaining_targets)]
        
        stats_output_path = os.path.join(output_dir, 'filtered_capri_quality_stats.csv')
        final_stats_df.to_csv(stats_output_path, index=False)
        print(f"Saved quality statistics for {len(final_stats_df)} remaining CAPRI targets to {stats_output_path}")


if __name__ == '__main__':
    # Override command line arguments for a specific run
    # This allows the script to be run without providing arguments manually
    sys.argv = [
        sys.argv[0], # The script name itself
        '--capri_root', '/scratch/jgray21/rzhu41/eudockscore_versus_dfmdock/capri_score_set_targets',
        '--capri_quality_dir', '/scratch/jgray21/rzhu41/eudockscore_versus_dfmdock/uploaders/database',
        '--dips_train', '/scratch/jgray21/rzhu41/DFMDock2/src/data/dips_train_hetero.txt',
        '--dips_val', '/scratch/jgray21/rzhu41/DFMDock2/src/data/dips_val_hetero.txt',
        '--output', '/scratch/jgray21/rzhu41/eudockscore_versus_dfmdock/filtered_capri_targets_list/nonoverlapping_with_dips_complexes.txt'
    ]
    
    main()