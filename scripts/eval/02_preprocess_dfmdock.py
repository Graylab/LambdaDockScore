#!/usr/bin/env python3
"""
Preprocess extracted PDB files into .pt files for DFMDock.

This script is adapted from DFMDock2's preprocessing scripts.
It takes the extracted decoy PDBs and generates graph-based .pt files.
"""
import argparse
import os
import torch

# Set TORCH_HOME to a directory on the scratch space to avoid filling up the home directory
# This ensures that the large model files are downloaded to a location with sufficient space.
torch_hub_dir = '/scratch/jgray21/rzhu41/.cache/torch'
os.environ['TORCH_HOME'] = torch_hub_dir

from tqdm import tqdm
import pandas as pd
from typing import Dict, Optional

# It's assumed that the environment has DFMDock's dependencies.
# We will dynamically add the DFMDock source to the path.
import sys
dfmdock_path = '/scratch/jgray21/rzhu41/DFMDock'
sys.path.insert(0, os.path.join(dfmdock_path, 'src'))
from DockDPO_data_gen import PDBToTorchConverter

# Global converter to avoid repeated initialization
converter: Optional[PDBToTorchConverter] = None


def ensure_converter_initialized() -> PDBToTorchConverter:
    global converter
    if converter is None:
        converter = PDBToTorchConverter()
    return converter


def process_pdb_file(pdb_path: str, output_dir: str, decoy_metadata: Dict[str, Dict]) -> Optional[str]:
    global converter
    try:
        decoy_filename = os.path.basename(pdb_path)
        
        # Find metadata for the current decoy
        matching_keys = [k for k in decoy_metadata if k.endswith(decoy_filename)]
        if not matching_keys:
            print(f"Warning: Metadata not found for {decoy_filename}")
            return None
        metadata = decoy_metadata[matching_keys[0]]

        target_id = metadata['target_id']
        
        # Define output path for the .pt file
        pt_filename = decoy_filename.replace('.pdb', '.pt')
        pt_output_dir = os.path.join(output_dir, target_id, 'pt')
        os.makedirs(pt_output_dir, exist_ok=True)
        pt_path = os.path.join(pt_output_dir, pt_filename)

        if os.path.exists(pt_path):
            return pt_path

        # Use the converter to create the HeteroData object
        # This will parse the PDB, get sequences, and generate ESM embeddings
        converter_instance = ensure_converter_initialized()
        data = converter_instance.convert_pdb_to_pt(pdb_path, output_path=None)
        if data is None:
            return None
            
        # Add metadata from the competition's data
        data.dockq = torch.tensor([metadata['dockq']], dtype=torch.float32)
        data.model_num = metadata['model_num']
        data.target_id = metadata['target_id']
        data.pdb_path = pdb_path

        # Save the final data object
        torch.save(data, pt_path)
        return pt_path
    except Exception as e:
        print(f"Error processing {pdb_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Preprocess PDBs for DFMDock.')
    parser.add_argument('--extracted_dir', type=str, required=True, help='Directory with extracted decoys (e.g., capri_decoys_extracted).')
    parser.add_argument('--metadata_file', type=str, required=True, help='Path to the decoy_metadata.json file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save .pt files.')
    args = parser.parse_args()

    with open(args.metadata_file, 'r') as f:
        decoy_metadata = pd.read_json(f, orient='index').T.to_dict()

    pdb_files = []
    for target_id in os.listdir(args.extracted_dir):
        target_dir = os.path.join(args.extracted_dir, target_id)
        if not os.path.isdir(target_dir):
            continue
        pdb_dir = os.path.join(target_dir, 'pdb')
        if os.path.isdir(pdb_dir):
            for pdb_file in os.listdir(pdb_dir):
                if pdb_file.endswith('.pdb'):
                    pdb_files.append(os.path.join(pdb_dir, pdb_file))

    print(f"Found {len(pdb_files)} PDB files to process.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize converter once before processing to surface issues early
    ensure_converter_initialized()

    results = []
    for pdb_path in tqdm(pdb_files, desc="Generating .pt files"):
        result = process_pdb_file(pdb_path, args.output_dir, decoy_metadata)
        if result:
            results.append(result)

    print(f"Successfully generated {len(results)} .pt files in {args.output_dir}")

if __name__ == '__main__':
    main()
