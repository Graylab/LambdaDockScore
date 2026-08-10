import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# import packages
import os
import csv
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils import data
from scipy.spatial.transform import Rotation 
from models.score_model_mlsb import Score_Model
from datasets.ppi_mlsb_dataset import PPIDataset
from utils.geometry import axis_angle_to_matrix, matrix_to_axis_angle
from utils.pdb import save_PDB, place_fourth_atom 
from utils.metrics import compute_metrics
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from pyrosetta_scripts.rosetta_pack_and_relax import calculate_complex_energies
from Ranking_Net import load_embedding_from_reference
import argparse
import json

data_dirs = [
    "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/311789_dips_hetero_model_0_db5_all/run0",
    "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/311790_dips_hetero_model_0_db5_all/run0",
    "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/311791_dips_hetero_model_0_db5_all/run0",
    "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/311792_dips_hetero_model_0_db5_all/run0"
]

def compute_ranking_results(data_dirs, model_checkpoint_path, energy_type, debug=False, debug_dataset="/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/ranking_expt_trial_run_sampled_50_poses_per_id.json", keep_full_id=False):
    # Collect all individual_poses.json files from data_dirs and subdirectories
    individual_poses_data = []
    for data_dir in data_dirs:
        for root, dirs, files in os.walk(data_dir):
            if 'individual_poses.json' in files:
                json_file_path = os.path.join(root, 'individual_poses.json')
                try:
                    with open(json_file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            individual_poses_data.extend(data)
                        else:
                            individual_poses_data.append(data)
                    print(f"Loaded {len(data) if isinstance(data, list) else 1} entries from {json_file_path}")
                except Exception as e:
                    print(f"Error reading {json_file_path}: {e}")

    print(f"Total individual poses data entries: {len(individual_poses_data)}")
    # Convert list to dictionary using 'path' as key
    individual_poses_dict = {}
    for entry in individual_poses_data:
        if 'path' in entry:
            individual_poses_dict[entry['path']] = entry
        else:
            print(f"Warning: Entry missing 'path' key: {entry}")

    # Collect and pair PDB and PT files
    pdb_pt_pairs = []

    if debug:
        with open(debug_dataset, 'r') as f:
            debug_data = json.load(f)

        for item in debug_data:
            pt_path = item['path']
            dockq = item['dockq']

            # Convert .pt path to corresponding .pdb path
            # Assuming the .pdb file is in a parallel directory structure
            base_name = os.path.splitext(os.path.basename(pt_path))[0]
            pdb_file = base_name + '.pdb'
            
            # Find the corresponding pdb directory
            pt_dir = os.path.dirname(pt_path)
            data_dir = os.path.dirname(pt_dir)  # Go up one level from 'pt' directory
            pdb_dir = os.path.join(data_dir, 'pdbs')
            pdb_path = os.path.join(pdb_dir, pdb_file)
            
            # Check if corresponding PDB file exists
            if os.path.exists(pdb_path):
                pdb_pt_pairs.append((pdb_path, pt_path, dockq))
            else:
                print(f"Warning: No corresponding PDB file found for {pt_path}")

    else:
        for data_dir in data_dirs:
            pdb_dir = os.path.join(data_dir, 'pdbs')
            pt_dir = os.path.join(data_dir, 'pt')
            
            # Check if both directories exist
            if not os.path.exists(pdb_dir) or not os.path.exists(pt_dir):
                print(f"Warning: Missing pdbs or pt directory in {data_dir}")
                continue
            
            # Get all PDB files in the pdbs directory
            pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
            
            for pdb_file in pdb_files:
                # Construct corresponding PT file name
                base_name = os.path.splitext(pdb_file)[0]
                pt_file = base_name + '.pt'
                
                pdb_path = os.path.join(pdb_dir, pdb_file)
                pt_path = os.path.join(pt_dir, pt_file)

                dockq = individual_poses_dict.get(pt_path, {}).get('dockq', None)
                
                # Check if corresponding PT file exists
                if os.path.exists(pt_path):
                    pdb_pt_pairs.append((pdb_path, pt_path, dockq))
                else:
                    print(f"Warning: No corresponding PT file found for {pdb_file}")

    print(f"Found {len(pdb_pt_pairs)} PDB-PT pairs")
    return run_parallel_pdb_pt_pairs(pdb_pt_pairs, model_checkpoint_path, energy_type, keep_full_id)

def run_parallel_pdb_pt_pairs(pdb_pt_pairs, model_checkpoint_path, energy_type, keep_full_id):
    ctx = mp.get_context('spawn')
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available, using CPU")
        num_workers = 1
    else:
        print(f"Found {num_gpus} GPUs, using all for parallel processing")
        num_workers = num_gpus
    
    if energy_type == 'Rosetta':
        # CPU‐only: spin up one process per core
        num_workers = mp.cpu_count()
        print(f"Rosetta mode: using {num_workers} CPU workers")
        num_gpus = 0
    else:
        # DFMDock mode: use all available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs available, falling back to 1 CPU worker")
            num_workers = 1
        else:
            print(f"Found {num_gpus} GPUs, using all for parallel processing")
            num_workers = num_gpus

    # Split pairs among workers
    chunk_size = len(pdb_pt_pairs) // num_workers + (1 if len(pdb_pt_pairs) % num_workers else 0)
    chunks = [pdb_pt_pairs[i:i + chunk_size] for i in range(0, len(pdb_pt_pairs), chunk_size)]

    # Process chunks in parallel
    metrics_list = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            gpu_id = i % num_gpus if num_gpus > 0 else None
            future = executor.submit(process_chunk, chunk, model_checkpoint_path, gpu_id, energy_type, keep_full_id)
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            chunk_metrics = future.result()
            metrics_list.extend(chunk_metrics)

    return metrics_list

def process_chunk(pdb_pt_pairs_chunk, model_checkpoint_path, gpu_id, energy_type, keep_full_id):
    """Process a chunk of PDB-PT pairs on a specific GPU"""
    # Set device for this process
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")

    # Load the dataset loader
    dataset_loader = PPIDataset(
        dataset=None
    )
    
    # Load model in this process
    model = Score_Model.load_from_checkpoint(
        model_checkpoint_path,
        map_location=device,
    )
    model.eval()
    model.to(device)
    
    chunk_metrics = []
    for idx, (pdb_path, pt_path, dockq) in enumerate(tqdm(pdb_pt_pairs_chunk, desc=f"GPU {gpu_id}")):
        pair_name = os.path.basename(pdb_path).replace('.pdb', '')
        metrics = compute_individual_ranking(pdb_path, pt_path, dockq, model, device, energy_type, dataset_loader, keep_full_id)
        chunk_metrics.append(metrics)
    
    return chunk_metrics

def compute_individual_ranking(pdb_path, pt_path, dockq, model, device, energy_type, dataset_loader, keep_full_id):
    # Load the PDB file
    if energy_type == 'Rosetta':
        _, _, energy = calculate_complex_energies(complex_pdb_file=pdb_path, num_fast_relax_repeats=50)
        num_clashes = 0  # Placeholder, as Rosetta does not provide clash count directly

    # Load the PT file
    raw_data = torch.load(pt_path, weights_only=False)
    if keep_full_id:
        _id = os.path.basename(pt_path).replace('.pt', '')
    else:
        _id = os.path.basename(pt_path).split('_')[0]
    if hasattr(raw_data, 'embedding_reference_info') and str(raw_data.embedding_reference_info).strip():
        rec_embedding, lig_embedding = load_embedding_from_reference(raw_data.embedding_reference_info)
        raw_data['receptor'].x = rec_embedding
        raw_data['ligand'].x = lig_embedding
    batch = dataset_loader.construct_full_output(raw_data=raw_data, _id=_id, swap=False)
    rec_x = batch['rec_x'].to(device).squeeze(0)
    lig_x = batch['lig_x'].to(device).squeeze(0)
    rec_pos = batch['rec_pos'].to(device).squeeze(0)
    lig_pos = batch['lig_pos'].to(device).squeeze(0)
    position_matrix = batch['position_matrix'].to(device).squeeze(0)

    if energy_type == 'DFMDock':
        batch = {
                    "rec_x": rec_x,
                    "lig_x": lig_x,
                    "rec_pos": rec_pos.clone().detach(),
                    "lig_pos": lig_pos.clone().detach(),
                    "position_matrix": position_matrix,
                }

        batch["t"] = torch.zeros(1, device=device) + 1e-5
        output = model(batch)
        energy = output["energy"].item()
        num_clashes = output["num_clashes"].item()

    metrics = {'id': _id}
    metrics.update({'DockQ': dockq})
    metrics.update({'energy': energy})
    metrics.update({'num_clashes': num_clashes})

    return metrics

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Compute ranking results using DFMDock or Rosetta energy')
    parser.add_argument('--energy_type', type=str, choices=['DFMDock', 'Rosetta'], 
                       default='DFMDock', help='Type of energy to use for ranking (default: DFMDock)')
    parser.add_argument('--debug', type=str, default="False",
                       help='Run in debug mode with limited data (default: False)')
    parser.add_argument('--model_checkpoint_path', type=str, default='checkpoints/test_ckpts/dips_hetero/308593_epoch_36-step_99197.ckpt')
    parser.add_argument('--slurm_id', type=str, default=None, help='SLURM job ID for logging purposes')
    parser.add_argument('--experiment', type=str, default='capri_score_set',)
    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)

    # Model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path

    # output directory
    if args.debug == "True":
        output_dir = f"/scratch/jgray21/rzhu41/DFMDock/src/DFMDock_vs_Rosetta_energy_ranking_expts_db5_all/csv_files_debug_{args.slurm_id}"
        debug_flag = True
    else:
        output_dir = f"/scratch/jgray21/rzhu41/DFMDock/src/DFMDock_vs_Rosetta_energy_ranking_expts_db5_all/csv_files_{args.slurm_id}"
        debug_flag = False

    os.makedirs(output_dir, exist_ok=True)

    # Compute ranking results
    output_filename = f'ranking_results_{args.energy_type.lower()}.csv'
    with open(os.path.join(output_dir, output_filename), 'w', newline='') as csvfile:
        if args.experiment == 'capri_score_set':
            # Load CAPRI score set data
            capri_data_path = "/scratch/jgray21/rzhu41/684/capri_score_set_dips_cleaned/capri_scoreset_eight_proteins_pdb_pt_pairs.json"
            with open(capri_data_path, 'r') as f:
                capri_data = json.load(f)
            
            # Convert to the expected format (pdb_path, pt_path, dockq)
            pdb_pt_pairs = []
            for item in capri_data:
                pdb_path = item[0]
                pt_path = item[1]
                dockq = item[2]
                if pdb_path and pt_path:
                    pdb_pt_pairs.append((pdb_path, pt_path, dockq))
            
            metrics_list = run_parallel_pdb_pt_pairs(pdb_pt_pairs, model_checkpoint_path, energy_type=args.energy_type, keep_full_id=True)
        else:
            metrics_list = compute_ranking_results(data_dirs, model_checkpoint_path, energy_type=args.energy_type, debug=debug_flag, keep_full_id=False)

        headers = list(metrics_list[0].keys()) if metrics_list else []
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for row in metrics_list:
            writer.writerow(row)
