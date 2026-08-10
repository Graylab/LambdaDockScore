#!/usr/bin/env python3
"""
Calculate and report ranking metrics for DFMDock and EuDockScore.

This script reads the score files from both models, merges them with ground truth
DockQ values, and computes top-k and oracle accuracy.
"""
import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_metrics(df, score_col, ground_truth_col, ascending=True):
    """
    Calculates top-k and oracle success rates.
    A success is defined as selecting a model with DockQ >= 0.23 (acceptable).
    """
    metrics = {}
    acceptable_threshold = 0.23
    
    # Group by target
    grouped = df.groupby('target_id')
    
    success_counts = defaultdict(int)
    oracle_success_counts = 0
    total_targets = 0

    for target_id, group in grouped:
        total_targets += 1
        
        # Sort by score
        sorted_group = group.sort_values(by=score_col, ascending=ascending)
        
        # Top-k success
        for k in [1, 5]:
            top_k = sorted_group.head(k)
            if (top_k[ground_truth_col] >= acceptable_threshold).any():
                success_counts[k] += 1
        
        # Oracle success (is there at least one acceptable model in the set?)
        if (group[ground_truth_col] >= acceptable_threshold).any():
            oracle_success_counts += 1
            
    if total_targets > 0:
        for k in [1, 5]:
            metrics[f'top_{k}_accuracy'] = success_counts[k] / total_targets
        metrics['oracle_accuracy'] = oracle_success_counts / total_targets
    else:
        for k in [1, 5]:
            metrics[f'top_{k}_accuracy'] = 0.0
        metrics['oracle_accuracy'] = 0.0
        
    metrics['total_targets'] = total_targets
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Calculate ranking metrics.')
    parser.add_argument('--dfmdock_scores', type=str, required=True, help='CSV file with DFMDock scores.')
    parser.add_argument('--eudockscore_scores_dir', type=str, required=True, help='Directory with EuDockScore CSV files.')
    parser.add_argument('--metadata_file', type=str, required=True, help='JSON file with decoy metadata.')
    args = parser.parse_args()

    # --- Load Data ---
    # DFMDock
    dfmdock_df = pd.read_csv(args.dfmdock_scores)
    
    # EuDockScore
    eudock_files = [os.path.join(args.eudockscore_scores_dir, f) for f in os.listdir(args.eudockscore_scores_dir) if f.endswith('.csv')]
    eudock_df = pd.concat([pd.read_csv(f) for f in eudock_files])
    eudock_df.rename(columns={'File': 'model_id', 'Prediction': 'eudockscore_score'}, inplace=True)
    # model_id in eudock is just the filename, need to match it with dfmdock
    eudock_df['model_id'] = eudock_df['model_id'].apply(lambda x: x.replace('.pdb', ''))


    # Metadata
    with open(args.metadata_file, 'r') as f:
        metadata = pd.read_json(f, orient='index')
    metadata['model_id'] = metadata.index.str.replace('.pdb', '')

    # --- Merge Data ---
    # Merge DFMDock with metadata
    merged_df = pd.merge(dfmdock_df, metadata[['model_id', 'dockq']], on='model_id', how='left', suffixes=('', '_meta'))
    # Merge with EuDockScore
    merged_df = pd.merge(merged_df, eudock_df[['model_id', 'eudockscore_score']], on='model_id', how='left')

    print("Merged DataFrame head:")
    print(merged_df.head())
    print(f"\nTotal models in merged data: {len(merged_df)}")
    
    # --- Calculate Metrics ---
    print("\n--- DFMDock Metrics ---")
    # Lower score is better for DFMDock
    dfmdock_metrics = calculate_metrics(merged_df.dropna(subset=['dfmdock_score']), 'dfmdock_score', 'dockq', ascending=True)
    for key, value in dfmdock_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n--- EuDockScore Metrics ---")
    # Higher score is better for EuDockScore
    eudockscore_metrics = calculate_metrics(merged_df.dropna(subset=['eudockscore_score']), 'eudockscore_score', 'dockq', ascending=False)
    for key, value in eudockscore_metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    main()
