#!/usr/bin/env python3
"""
Score decoys using a DFMDock model.

This script is an adaptation of `ranking_model_comparisons.py`.
It loads pre-generated .pt files, scores them in parallel on available GPUs,
and saves the results to a CSV file.
"""
import argparse
import os
import sys
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# Add DFMDock and EuDockScore repos to path
dfmdock_path = os.environ.get('DFMDOCK_SRC', '../DFMDock')
sys.path.insert(0, os.path.join(dfmdock_path, 'src'))

from models.score_model_mlsb import Score_Model
from datasets.ppi_mlsb_dataset import PPIDataset

def score_chunk(pt_files_chunk, model_checkpoint_path, gpu_id, score_column_name):
    """Process a chunk of .pt files on a specific GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    dataset_loader = PPIDataset(dataset=None, training=False)
    
    model = Score_Model.load_from_checkpoint(
        model_checkpoint_path,
        map_location=device,
    )
    model.eval()
    model.to(device)
    
    chunk_results = []
    for pt_path in tqdm(pt_files_chunk, desc=f"GPU {gpu_id}", position=gpu_id, leave=False):
        try:
            raw_data = torch.load(pt_path, map_location='cpu', weights_only=False)
            _id = os.path.basename(pt_path).replace('.pt', '')

            batch = dataset_loader.construct_full_output(raw_data=raw_data, _id=_id, swap=False)
            
            # Move batch to device
            for key, value in batch.items():
                if hasattr(value, 'to'):
                    batch[key] = value.to(device)

            with torch.no_grad():
                batch["t"] = torch.zeros(1, device=device) + 1e-5
                output = model(batch)
                energy = output["energy"].item()
                num_clashes = output["num_clashes"].item()

            result = {
                'model_id': _id,
                'target_id': raw_data.target_id,
                score_column_name: energy,
                'num_clashes': num_clashes,
                'dockq': raw_data.dockq.item()
            }
            chunk_results.append(result)
        except Exception as e:
            print(f"Error scoring {pt_path} on GPU {gpu_id}: {e}")
            continue
            
    return chunk_results

def run_scoring(pt_files, model_checkpoint, output_csv, score_column_name, num_gpus):
    """Helper function to run the scoring process for a single model."""
    print(f"Scoring with model: {os.path.basename(model_checkpoint)}")
    
    chunk_size = len(pt_files) // num_gpus + (1 if len(pt_files) % num_gpus > 0 else 0)
    chunks = [pt_files[i:i + chunk_size] for i in range(0, len(pt_files), chunk_size)]

    available_gpus = list(range(num_gpus))
    if not available_gpus:
        raise RuntimeError("No GPUs available for scoring.")

    all_results = []
    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Scoring chunks")):
        if not chunk:
            continue

        gpu_id = available_gpus[chunk_idx % len(available_gpus)]
        print(f"Processing chunk {chunk_idx + 1}/{len(chunks)} on GPU {gpu_id} ({len(chunk)} decoys)")
        chunk_results = score_chunk(chunk, model_checkpoint, gpu_id, score_column_name)
        all_results.extend(chunk_results)

    df = pd.DataFrame(all_results)
    # If the dataframe contains no meaningful values (all-null or all-empty strings / no columns),
    # skip writing and return early.
    meaningful_mask = df.notnull() & ~df.applymap(lambda x: isinstance(x, str) and x.strip() == "")
    if not meaningful_mask.values.any():
        print(f"No meaningful data to write for {output_csv}; skipping.")
        return
    df.to_csv(output_csv, index=False)
    print(f"Scoring complete. Results saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Score decoys with DFMDock.')
    parser.add_argument('--pt_dir', type=str, required=True, help='Directory containing the .pt files.')
    parser.add_argument('--baseline_checkpoint', type=str, required=True, help='Path to the baseline DFMDock model checkpoint.')
    parser.add_argument('--finetuned_checkpoint', type=str, required=True, help='Path to the fine-tuned DFMDock model checkpoint.')
    parser.add_argument('--output_csv', type=str, required=True, help='Base path to save the output CSV files.')
    parser.add_argument('--target_list', type=str, required=True, help='Path to a file containing the list of target names (one per line).')
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    # Read target list
    with open(args.target_list, 'r') as f:
        targets = [line.strip() for line in f if line.strip()]

    pt_files = []
    for target in targets:
        target_dir = os.path.join(args.pt_dir, target)
        if not os.path.isdir(target_dir):
            print(f"Warning: Target directory {target_dir} does not exist, skipping.")
            continue
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith('.pt'):
                    pt_files.append(os.path.join(root, file))
    
    print(f"Found {len(pt_files)} .pt files to score.")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        sys.exit("No GPUs available. This script requires at least one GPU.")
    
    print(f"Using {num_gpus} GPUs for scoring.")

    # Define output paths
    output_base = args.output_csv.rsplit('.csv', 1)[0]
    baseline_output_csv = f"{output_base}_baseline.csv"
    finetuned_output_csv = f"{output_base}_finetuned.csv"

    # Run scoring for the baseline model
    run_scoring(pt_files, args.baseline_checkpoint, baseline_output_csv, 'dfmdock_score_baseline', num_gpus)

    # Run scoring for the fine-tuned model
    run_scoring(pt_files, args.finetuned_checkpoint, finetuned_output_csv, 'dfmdock_score_finetuned', num_gpus)


if __name__ == '__main__':
    main()
