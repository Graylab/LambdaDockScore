import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import csv
import json
import torch
import torch.multiprocessing as mp
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
from inference_mlsb_Score_Net import sample_sphere, rot_compose, get_full_coords
import random
import glob
import esm
from collections import defaultdict
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1
import esm
from torch_geometric.data import HeteroData
from omegaconf import OmegaConf
import seaborn as sns

#----------------------------------------------------------------------------
# Data class for pose

@dataclass
class Pose:
    _id: str
    rec_seq: str
    lig_seq: str
    rec_pos: torch.FloatTensor
    lig_pos: torch.FloatTensor
    dockq: float
    index: str = None
    pose_type: str = None  # 'inference' or 'perturbed_gt'
    unique_str_id: str = None  # Unique identifier for the pose

#----------------------------------------------------------------------------
# Helper functions

def save_pose_pdb(pose, output_path):
    """Save a pose to PDB file"""
    if os.path.exists(output_path):
        os.remove(output_path)
        
    coords = torch.cat([pose.rec_pos, pose.lig_pos], dim=0)
    coords = get_full_coords(coords)
    seq = pose.rec_seq + pose.lig_seq
    
    save_PDB(out_pdb=output_path, coords=coords, seq=seq, delim=len(pose.rec_seq)-1)

def compute_dockq(pred_coords, gt_coords):
    """Compute DockQ score between predicted and ground truth coordinates"""
    metrics = compute_metrics(pred_coords, gt_coords)
    return metrics.get('DockQ')

def apply_small_perturbation(coords, max_translation=1.0, max_rotation=8.0, tr_type='normal', rot_type='normal'):
    """Apply small random perturbation to coordinates
    
    Args:
        coords: Input coordinates
        max_translation: Maximum translation magnitude
        max_rotation: Maximum rotation angle in degrees
        tr_type: 'normal' or 'uniform' distribution for translation
        rot_type: 'normal' or 'uniform' distribution for rotation
    """
    coords = coords.clone()
    
    # Get center of mass
    center = torch.mean(coords[..., 1, :], dim=0, keepdim=True)  # Use CA atoms
    
    # Random rotation (small)
    if rot_type == 'uniform':
        rot_angle = (torch.rand(1, device=coords.device) * 2 - 1) * max_rotation * torch.pi / 180  # in radians
    elif rot_type == 'normal':  # normal
        rot_angle = torch.randn(1, device=coords.device) * max_rotation * torch.pi / 180  # in radians
        
    rot_axis = torch.randn(3, device=coords.device)
    rot_axis = rot_axis / torch.norm(rot_axis)
    rot_vec = rot_angle * rot_axis
    rot_matrix = axis_angle_to_matrix(rot_vec.unsqueeze(0)).squeeze(0)
    
    # Random translation (small)
    if tr_type == 'uniform':
        translation = (torch.rand(1, 3, device=coords.device) * 2 - 1) * max_translation
    elif tr_type == 'normal':  # normal
        translation = torch.randn(1, 3, device=coords.device) * max_translation
    
    # Apply perturbation
    coords = (coords - center) @ rot_matrix.T + center + translation
    
    return coords

#----------------------------------------------------------------------------
# PDB to Torch converter
#----------------------------------------------------------------------------
# DPO Utils
class PDBToTorchConverter:
    """
    Utility class to convert PDB files to .pt files with ESM embeddings.
    """
    def __init__(self, esm_model_name='esm2_t33_650M_UR50D'):
        # Load ESM model from online HuggingFace repository
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.esm_model = self.esm_model.to(self.device).eval()
        
        # Amino acid code mapping
        self.aa_code = defaultdict(lambda: "<unk>")
        self.aa_code.update({k.upper(): v for k, v in protein_letters_3to1.items()})
        
        # PDB parser
        self.parser = PDBParser()
        
    def get_esm_rep(self, seq_prim):
        """
        Get ESM representations for a sequence using ESM2 650M model.
        Updated method based on the reference code.
        """
        seq = [("seq", seq_prim)]
        out = self.batch_converter(seq)
        with torch.no_grad():
            results = self.esm_model(out[-1].to(self.device), repr_layers=[33])
            rep = results["representations"][33].cpu()
        return rep[0, 1:-1, :]  # Remove start and end tokens
    
    def parse_pdb_file(self, pdb_path, custom_chain_ids=None):
        """
        Parse PDB file and extract coordinates and sequences for receptor (chain A) and ligand (chain B).
        """
        structure = self.parser.get_structure('complex', pdb_path)
        
        receptor_data = {'pos': [], 'res': []}
        ligand_data = {'pos': [], 'res': []}
        
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                
                if custom_chain_ids:
                    receptor_id = custom_chain_ids.get('receptor', 'A')
                    ligand_id = custom_chain_ids.get('ligand', 'B')
                else:
                    receptor_id = 'A'
                    ligand_id = 'B'

                if chain_id == receptor_id:  # Receptor
                    target_data = receptor_data
                elif chain_id == ligand_id:  # Ligand
                    target_data = ligand_data
                else:
                    print("Warning: Unexpected chain ID:", chain_id)
                    continue
                
                for residue in chain:
                    if residue.get_id()[0] != ' ':  # Skip hetero residues
                        continue
                    
                    res_name = residue.get_resname()
                    target_data['res'].append(res_name)

                    # Extract backbone coordinates (N, CA, C)
                    coords = []
                    for atom_name in ['N', 'CA', 'C']:
                        if atom_name in residue:
                            coords.append(residue[atom_name].get_coord())
                        else:
                            print(f"Warning: {atom_name} not found in residue {residue.get_id()} of chain {chain_id} in {pdb_path}")
                    
                    target_data['pos'].append(coords) #we want the coordinates to just be of the N, Ca, C atoms, no O nor Cb
        
        return receptor_data, ligand_data
    
    def convert_pdb_to_pt(self, pdb_path, output_path, esm_embeddings_provided=False, file_path=None, custom_receptor_ligand_chain_id=None):
        """
        Convert a PDB file to .pt format with ESM embeddings.
        If ESM embeddings are provided, they can be loaded in from the file_path specified, which must not be None.
        """
        # Parse PDB file
        receptor_data, ligand_data = self.parse_pdb_file(pdb_path, custom_receptor_ligand_chain_id)
        
        # Convert amino acid codes from 3-letter to 1-letter
        receptor_seq = "".join(self.aa_code[res] for res in receptor_data['res'])
        ligand_seq = "".join(self.aa_code[res] for res in ligand_data['res'])
        
        # Get ESM embeddings
        if esm_embeddings_provided:
            if file_path is None:
                raise ValueError("File path must be provided when ESM embeddings are provided.")
            receptor_esm = torch.empty(0)
            ligand_esm = torch.empty(0)
            file_path = os.path.abspath(file_path)
        else:
            receptor_esm = self.get_esm_rep(receptor_seq)
            ligand_esm = self.get_esm_rep(ligand_seq)
            file_path = "" # No file path needed if ESM embeddings are computed
        
        # Convert coordinates to torch tensors
        receptor_coords = torch.tensor(receptor_data['pos'], dtype=torch.float32)
        ligand_coords = torch.tensor(ligand_data['pos'], dtype=torch.float32)
        
        # Create HeteroData object
        data = HeteroData()
        data.name = os.path.basename(pdb_path).replace('.pdb', '')
        data['receptor'].x = receptor_esm
        data['receptor'].pos = receptor_coords
        data['receptor'].seq = receptor_seq
        data['ligand'].x = ligand_esm
        data['ligand'].pos = ligand_coords
        data['ligand'].seq = ligand_seq
        data.embedding_reference_info = str(file_path)  # Store reference path

        # Validate dimensions
        if not esm_embeddings_provided:
            assert (receptor_esm.size(0) == receptor_coords.size(0) and 
                    receptor_coords.size(0) == len(receptor_seq) and
                    ligand_esm.size(0) == ligand_coords.size(0) and 
                    ligand_coords.size(0) == len(ligand_seq))
        
        if output_path is not None:
            torch.save(data, output_path)

        return data

#----------------------------------------------------------------------------
# DPO Dataset Generator

class DPODatasetGenerator:
    def __init__(self, config: DictConfig):
        self.data_conf = config.data
        self.world_size  = config.distributed.world_size
        self.rank      = config.distributed.rank
        self.output_dir = self.data_conf.output_dir
        self.rank_dir = os.path.join(self.output_dir, f"rank_{self.rank}")
        os.makedirs(self.rank_dir, exist_ok=True)
        self.pairs_file = os.path.join(self.rank_dir, "preference_pairs.json")
        self.individual_poses_file = os.path.join(self.rank_dir, "individual_poses.json")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = Score_Model.load_from_checkpoint(
            self.data_conf.ckpt, 
            map_location=self.device,
        )
        self.model.eval()
        self.model.to(self.device)
        
        # Load dataset
        self.dataset = PPIDataset(
            dataset=self.data_conf.dataset, 
            training=False, 
            use_esm=self.data_conf.use_esm,
        )
        
        # Create output directories
        self.pdb_dir = os.path.join(self.output_dir, 'pdbs')
        self.pt_dir = os.path.join(self.output_dir, 'pt')
        os.makedirs(self.pdb_dir, exist_ok=True)
        os.makedirs(self.pt_dir, exist_ok=True)
        
        # Initialize PDB to torch converter
        self.pdb_converter = PDBToTorchConverter()
        
        # Storage for all poses
        self.all_poses = []
        self.preference_pairs = []
        self.individual_poses = []
        
    def load_existing_pairs(self):
        """Load existing preference pairs if the file exists"""
        if os.path.exists(self.pairs_file):
            try:
                with open(self.pairs_file, 'r') as f:
                    self.preference_pairs = json.load(f)
                print(f"Loaded {len(self.preference_pairs)} existing preference pairs")
            except:
                print("Could not load existing pairs file, starting fresh")
                self.preference_pairs = []
        else:
            self.preference_pairs = []
            
        if os.path.exists(self.individual_poses_file):
            try:
                with open(self.individual_poses_file, 'r') as f:
                    self.individual_poses = json.load(f)
                print(f"Loaded {len(self.individual_poses)} existing individual poses")
            except:
                print("Could not load existing individual poses file, starting fresh")
                self.individual_poses = []
        else:
            self.individual_poses = []
    
    def save_preference_pairs(self):
        """Save preference pairs to JSON file"""
        with open(self.pairs_file, 'w') as f:
            json.dump(self.preference_pairs, f, indent=2)
            
    def save_individual_poses(self):
        """Save individual poses to JSON file"""
        with open(self.individual_poses_file, 'w') as f:
            json.dump(self.individual_poses, f, indent=2)
        
    def add_individual_pose(self, pose, pt_path):
        """Add a pose to the individual poses JSON file, avoiding duplicates"""
        pose_entry = {
            "path": pt_path,
            "generation_method": pose.pose_type,
            "id": pose._id,
            "sample_number": int(pose.index.split('_')[-1]),
            "dockq": pose.dockq,
            "unique_str_id": pose.unique_str_id,
        }
        
        # Check if entry already exists
        for existing_pose in self.individual_poses:
            if existing_pose["path"] == pose_entry["path"]:
                return  # Don't add duplicate
                
        self.individual_poses.append(pose_entry)
        
    def generate_ground_truth_pose(self, batch):
        """Generate ground truth pose with no perturbation"""
        pose = Pose(
            _id=batch['id'],
            rec_seq=batch['rec_seq'],
            lig_seq=batch['lig_seq'],
            rec_pos=batch['rec_pos_original'],
            lig_pos=batch['lig_pos_original'],
            dockq=1.0,  # Perfect DockQ for ground truth
            index="gt_0",
            pose_type="ground_truth",
            unique_str_id=''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=16))
        )
        return pose
        
    def run_inference_sampling(self, batch, num_samples=10, target_dockq=None, max_attempts=None):
        """Run inference sampling to generate multiple poses"""
        poses = []

        if target_dockq is not None and max_attempts is not None:
            attempts = 0

        while len(poses) < num_samples:
            rec_trj, lig_trj, energy = self.euler_maruyama_sampler(batch)
            
            # Get final coordinates
            final_rec_pos = rec_trj[-1]
            final_lig_pos = lig_trj[-1]
            
            # Compute DockQ
            gt_coords = [batch['rec_pos_original'], batch['lig_pos_original']]
            pred_coords = [final_rec_pos, final_lig_pos]
            dockq = compute_dockq(pred_coords, gt_coords)
            
            pose = Pose(
                _id=batch['id'],
                rec_seq=batch['rec_seq'],
                lig_seq=batch['lig_seq'],
                rec_pos=final_rec_pos,
                lig_pos=final_lig_pos,
                dockq=dockq,
                index=f"inf_{len(poses)}",
                pose_type="inference",
                unique_str_id=''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=16))
            )

            if target_dockq is not None and max_attempts is not None:
                if dockq > target_dockq:
                    poses.append(pose)
                attempts += 1
                if attempts >= max_attempts:
                    print(f"Max attempts reached: {max_attempts}. Stopping inference sampling.")
                    break
            else:
                poses.append(pose)

        return poses
    
    def generate_perturbed_gt_samples(self, batch, num_samples=10, target_dockq=0.23, max_attempts=200, max_rot=8.0, max_translation=1.0):
        """Generate perturbed ground truth samples with DockQ > target_dockq"""
        poses = []
        attempts = 0
        
        while len(poses) < num_samples and attempts < max_attempts:
            # Apply small perturbation to ground truth
            perturbed_lig_pos = apply_small_perturbation(
                batch['lig_pos_original'],
                max_translation=max_translation,
                max_rotation=max_rot,
                tr_type=self.data_conf.tr_type,
                rot_type=self.data_conf.rot_type
            )
            
            # Compute DockQ
            gt_coords = [batch['rec_pos_original'], batch['lig_pos_original']]
            pred_coords = [batch['rec_pos_original'], perturbed_lig_pos]
            dockq = compute_dockq(pred_coords, gt_coords)
            
            if dockq > target_dockq:
                pose = Pose(
                    _id=batch['id'],
                    rec_seq=batch['rec_seq'],
                    lig_seq=batch['lig_seq'],
                    rec_pos=batch['rec_pos_original'],
                    lig_pos=perturbed_lig_pos,
                    dockq=dockq,
                    index=f"gt_{len(poses)}",
                    pose_type="perturbed_gt",
                    unique_str_id=''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=16))
                )
                poses.append(pose)

            attempts += 1
            
        return poses
    
    def euler_maruyama_sampler(self, batch):
        """Euler-Maruyama sampler for inference"""
        rec_trj = []
        lig_trj = []
        
        batch_size = 1
        eps = 1e-3
        
        # Initialize time steps
        time_steps = torch.linspace(1., eps, self.data_conf.num_steps, device=self.device)
        dt = time_steps[0] - time_steps[1]
        
        # Get initial pose
        rec_pos = batch["rec_pos"] 
        lig_pos = batch["lig_pos"]
        
        # Randomly initialize coordinates
        rec_pos, lig_pos = self.randomize_pose(rec_pos, lig_pos)
        
        # Save initial coordinates 
        rec_trj.append(rec_pos)
        lig_trj.append(lig_pos)
        
        # Run reverse SDE
        with torch.no_grad():
            for i, time_step in enumerate(time_steps):
                is_last = i == time_steps.size(0) - 1
                t = torch.ones(batch_size, device=self.device) * time_step
                
                current_batch = {
                    "t": t,
                    "rec_x": batch["rec_x"],
                    "lig_x": batch["lig_x"],
                    "rec_pos": rec_pos.clone().detach(),
                    "lig_pos": lig_pos.clone().detach(),
                    "position_matrix": batch["position_matrix"],
                }
                
                # Get predictions
                output = self.model(current_batch)
                
                if not is_last:
                    tr_noise_scale = self.data_conf.tr_noise_scale
                    rot_noise_scale = self.data_conf.rot_noise_scale
                else:
                    tr_noise_scale = 0.0
                    rot_noise_scale = 0.0
                
                # Get rotation and translation updates
                rot = self.model.so3_diffuser.torch_reverse(
                    score_t=output["rot_score"].detach(),
                    t=t.item(),
                    dt=dt,
                    noise_scale=rot_noise_scale,
                    ode=self.data_conf.ode,
                )
                
                tr = self.model.r3_diffuser.torch_reverse(
                    score_t=output["tr_score"].detach(),
                    t=t.item(),
                    dt=dt,
                    noise_scale=tr_noise_scale,
                    ode=self.data_conf.ode,
                )
                
                # Apply updates
                lig_pos = self.modify_coords(lig_pos, rot, tr)
                
                # Save coordinates
                rec_trj.append(rec_pos)
                lig_trj.append(lig_pos)
                
        return rec_trj, lig_trj, output.get("energy", 0.0)
    
    def randomize_pose(self, x1, x2):
        """Randomize initial pose"""
        # x1, x2 have shape (batch_size, num_atoms, 3, 3)
        # Get center of mass
        c1 = torch.mean(x1[..., 1, :], dim=0, keepdim=True)
        c2 = torch.mean(x2[..., 1, :], dim=0, keepdim=True)
        
        # Random rotation
        rot_update = torch.from_numpy(Rotation.random().as_matrix()).float().to(self.device)
        
        # Random translation
        tr_update = torch.normal(0.0, 30.0, size=(1, 3), device=self.device)
        
        # Apply transformations
        x1 = x1 - c1.unsqueeze(0)
        x2 = x2 - c2.unsqueeze(0)
        x2 = x2 @ rot_update.T + tr_update
        
        return x1, x2
    
    def modify_coords(self, x, rot, tr):
        """Apply rotation and translation to coordinates"""
        center = torch.mean(x[..., 1, :], dim=0, keepdim=True)
        rot_matrix = axis_angle_to_matrix(rot).squeeze()
        
        # Apply rotation around center
        x = (x - center) @ rot_matrix.T + center
        
        # Apply translation
        x = x + tr
        
        return x
    
    def create_preference_pairs(self, poses):
        """Create preference pairs from poses"""
        # Sort poses by DockQ
        poses_sorted = sorted(poses, key=lambda p: p.dockq)
        
        # Create pairs: high DockQ vs low DockQ
        pairs = []
        high_dockq_poses = [p for p in poses_sorted if p.dockq > self.data_conf.target_dockq]
        low_dockq_poses = [p for p in poses_sorted if p.dockq <= self.data_conf.target_dockq]

        # Create all possible pairs between high and low DockQ poses
        for high_pose in high_dockq_poses:
            # Randomly sample low DockQ poses for pairing
            num_pairs = min(len(low_dockq_poses), self.data_conf.pairs_per_high_dockq_sample)
            sampled_low_poses = random.sample(low_dockq_poses, num_pairs)
            for low_pose in sampled_low_poses:
                pairs.append({
                    'winner': high_pose,
                    'loser': low_pose
                })
            
        return pairs
    
    def save_poses_and_pairs(self, poses, protein_id, file_path):
        """Save poses to PDB and PT files and create preference pairs"""
        saved_poses = []
        
        for pose in poses:
            # Create filenames
            pdb_filename = f"{protein_id}_{pose.index}_{pose.unique_str_id}.pdb"
            pt_filename = f"{protein_id}_{pose.index}_{pose.unique_str_id}.pt"
            pdb_filepath = os.path.join(self.pdb_dir, pdb_filename)
            pt_filepath = os.path.join(self.pt_dir, pt_filename)
            
            # Save PDB
            save_pose_pdb(pose, pdb_filepath)
            
            # Save PT file with ESM embeddings
            self.pdb_converter.convert_pdb_to_pt(
                pdb_filepath, 
                pt_filepath, 
                esm_embeddings_provided=True,
                file_path=file_path
            )
            
            # Update pose with filepaths
            pose.filepath = pdb_filepath
            pose.pt_filepath = pt_filepath
            saved_poses.append(pose)
            
            # Add to individual poses
            self.add_individual_pose(pose, pt_filepath)

        if self.data_conf.generate_pairs:
            # Create preference pairs
            pairs = self.create_preference_pairs(saved_poses)
            
            # Convert to required format and add to existing pairs
            for pair in pairs:
                self.preference_pairs.append({
                    'id': protein_id,
                    'winner_path': pair['winner'].pt_filepath,
                    'loser_path': pair['loser'].pt_filepath,
                    'winner_dockq': pair['winner'].dockq,
                    'loser_dockq': pair['loser'].dockq,
                    'winner_type': pair['winner'].pose_type,
                    'loser_type': pair['loser'].pose_type
                })
        
        # Save updated files after each protein
        self.save_preference_pairs()
        self.save_individual_poses()
        
        return saved_poses
    
    def run_full_pipeline(self):
        """Run the complete DPO dataset generation pipeline"""
        print("Starting DPO dataset generation...")
        
        # Load existing pairs if any
        self.load_existing_pairs()
        
        # Process each protein in the dataset
        # split the protein‐IDs evenly across GPUs
        all_idx = list(range(len(self.dataset)))
        
        # Sample K proteins if specified
        if self.data_conf.sample_pose_set != -1:
            # Set seed for reproducible sampling across ranks
            random.seed(42)
            all_idx = random.sample(all_idx, min(self.data_conf.sample_pose_set, len(all_idx)))
        elif (self.data_conf.pose_set_subset_indices[1] != -1) and (self.data_conf.pose_set_subset_indices[0] != -1):
            all_idx = all_idx[self.data_conf.pose_set_subset_indices[0]:self.data_conf.pose_set_subset_indices[1]]

        # Split the (possibly sampled) indices across GPUs
        local_idx = all_idx[self.rank::self.world_size]
        for idx in tqdm(local_idx, desc=f"GPU {self.rank}/{self.world_size}"):
            # Get data
            data = self.dataset[idx]
            
            # Get ESM embeddings once per protein complex (efficient)
            receptor_esm = data['rec_x'].squeeze(0)
            ligand_esm = data['lig_x'].squeeze(0)
            file_path = data['file_path']

            # Prepare batch
            batch = {
                'id': data['id'],
                'rec_seq': data['rec_seq'],
                'lig_seq': data['lig_seq'],
                'rec_x': receptor_esm.to(self.device),
                'lig_x': ligand_esm.to(self.device),
                'rec_pos': data['rec_pos'].squeeze(0).to(self.device),
                'lig_pos': data['lig_pos'].squeeze(0).to(self.device),
                'rec_pos_original': data['rec_pos'].squeeze(0).to(self.device),
                'lig_pos_original': data['lig_pos'].squeeze(0).to(self.device),
                'position_matrix': data['position_matrix'].squeeze(0).to(self.device),
            }
            
            print(f"\nProcessing protein {data['id']}...")

            # List to store all poses for this protein
            all_poses_protein = []

            if self.data_conf.generate_ground_truth:
                # Generate ground truth pose
                gt_pose = self.generate_ground_truth_pose(batch)
                all_poses_protein.append(gt_pose)
            
            inference_poses = []
            if self.data_conf.generate_inference:
                # Generate inference poses
                inference_poses = self.run_inference_sampling(batch, num_samples=self.data_conf.num_inference_samples, target_dockq=self.data_conf.inference_target_dockq, max_attempts=self.data_conf.max_inference_attempts)
                all_poses_protein.extend(inference_poses)
            
            perturbed_gt_poses = []
            if self.data_conf.generate_perturbed_gt:
                # Generate perturbed GT samples
                for i in range(len(self.data_conf.num_perturbation_samples)):
                    samples = self.data_conf.num_perturbation_samples[i]
                    max_rot = self.data_conf.max_rotation[i]
                    max_translation = self.data_conf.max_translation[i]
                    current_perturbed_poses = self.generate_perturbed_gt_samples(batch, num_samples=samples, target_dockq=self.data_conf.target_dockq, max_attempts=self.data_conf.max_perturbation_attempts, max_rot=max_rot, max_translation=max_translation)
                    perturbed_gt_poses.extend(current_perturbed_poses)
                print(f"THE LENGTH OF PERTURBED GT POSES: {len(perturbed_gt_poses)}")
                all_poses_protein.extend(perturbed_gt_poses)

            if len(all_poses_protein) > 0:
                # Save poses and create pairs (this will save/update the JSON)
                saved_poses = self.save_poses_and_pairs(all_poses_protein, data['id'], file_path=file_path)
                self.all_poses.extend(saved_poses)
                
                print(f"Generated 1 ground truth, {len(inference_poses)} inference poses and {len(perturbed_gt_poses)} perturbed GT poses")
                print(f"Created {len([p for p in self.preference_pairs if p['id'] == data['id']])} preference pairs")
                print(f"Updated preference pairs JSON with {len(self.preference_pairs)} total pairs")
            else:
                print(f"No valid poses generated for {data['id']}")
        
        # Save final results
        self.save_preference_pairs()
        self.save_individual_poses()
        
        print(f"\nDPO dataset generation complete!")

#----------------------------------------------------------------------------
# Main function

def _worker(rank, world_size, cfg):
    cfg.distributed = {"world_size": world_size, "rank": rank}
    torch.cuda.set_device(rank)
    DPODatasetGenerator(cfg).run_full_pipeline()

@hydra.main(version_base=None,
            config_path="/scratch/jgray21/rzhu41/DFMDock/configs",
            config_name="dpo_data_gen_config")
def main(cfg: DictConfig):
    print("---- FULL CONFIG ----")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    # now cfg.data.output_dir includes your SLURM override
    print(f"Using output_dir = {cfg.data.output_dir}")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    num_gpus = torch.cuda.device_count()

    # spawn as many GPUs as you have
    mp.spawn(_worker,
             args=(num_gpus, cfg),
             nprocs=num_gpus,
             join=True)
    
    merged_pairs = []
    output_dir = cfg.data.output_dir  # This should already be in your config

    for fname in glob.glob(os.path.join(output_dir, "rank_*/preference_pairs.json")):
        with open(fname) as f:
            merged_pairs += json.load(f)

    final_path = os.path.join(output_dir, "preference_pairs.json")
    with open(final_path, "w") as f:
        json.dump(merged_pairs, f, indent=2)

    print(f"✅ Merged {len(merged_pairs)} pairs into {final_path}")

    # Save summary statistics
    # Calculate statistics from merged pairs
    stats = {
        'total_pairs': len(merged_pairs),
        'proteins_processed': len(set(p['id'] for p in merged_pairs)),
        'avg_dockq_winners': np.mean([p['winner_dockq'] for p in merged_pairs]) if merged_pairs else 0,
        'avg_dockq_losers': np.mean([p['loser_dockq'] for p in merged_pairs]) if merged_pairs else 0,
    }
    
    stats_file = os.path.join(output_dir, 'generation_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    print(f"Preference pairs: {final_path}")
    print(f"Statistics: {stats_file}")

def create_preference_pairs(winner_poses_list, loser_poses_list, output_filename, 
                          winner_methods=None, winner_min_threshold=None, winner_max_threshold=None, 
                          loser_methods=None, loser_min_threshold=None, loser_max_threshold=None):
    """
    Create preference pairs between winner and loser poses for DPO training.
    This function reads poses from multiple JSON files, filters them based on
    generation methods and DockQ thresholds, and creates preference pairs.
    
    Args:
        winner_poses_list (list): List of file paths to JSON files containing winner poses
        loser_poses_list (list): List of file paths to JSON files containing loser poses
        output_filename (str): Name of output file (will be saved in DockDPO directory)
        winner_methods (list, optional): List of generation methods to include for winners
        winner_threshold (float, optional): Minimum DockQ threshold for winner poses
        loser_methods (list, optional): List of generation methods to include for losers
        loser_threshold (float, optional): Minimum DockQ threshold for loser poses (Note: MINIMUM!)
    
    Returns:
        list: A list of preference pair dictionaries
    """
    
    # Base output directory
    base_dir = "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO"
    output_path = os.path.join(base_dir, output_filename)
    
    # Read and concatenate all winner poses
    all_winner_poses = []
    for poses_path in winner_poses_list:
        with open(poses_path, 'r') as f:
            poses = json.load(f)
        all_winner_poses.extend(poses)
    
    # Read and concatenate all loser poses
    all_loser_poses = []
    for poses_path in loser_poses_list:
        with open(poses_path, 'r') as f:
            poses = json.load(f)
        all_loser_poses.extend(poses)
    
    # Filter winner poses
    winner_poses = all_winner_poses
    if winner_methods is not None:
        winner_poses = [
            pose for pose in winner_poses 
            if pose['generation_method'] in winner_methods
        ]
    if winner_min_threshold is not None:
        winner_poses = [
            pose for pose in winner_poses 
            if pose['dockq'] >= winner_min_threshold
        ]
    if winner_max_threshold is not None:
        winner_poses = [
            pose for pose in winner_poses 
            if pose['dockq'] <= winner_max_threshold
        ]
    
    # Filter loser poses
    loser_poses = all_loser_poses
    if loser_methods is not None:
        loser_poses = [
            pose for pose in loser_poses 
            if pose['generation_method'] in loser_methods
        ]
    if loser_min_threshold is not None:
        loser_poses = [
            pose for pose in loser_poses 
            if pose['dockq'] >= loser_min_threshold
        ]
    if loser_max_threshold is not None:
        loser_poses = [
            pose for pose in loser_poses 
            if pose['dockq'] <= loser_max_threshold
        ]
    
    # Group loser poses by id
    loser_by_id = {}
    for pose in loser_poses:
        protein_id = pose['id']
        if protein_id not in loser_by_id:
            loser_by_id[protein_id] = []
        loser_by_id[protein_id].append(pose)
    
    # Create preference pairs
    preference_pairs = []
    
    for winner_pose in winner_poses:
        protein_id = winner_pose['id']
        
        # Find matching loser poses for this protein
        if protein_id in loser_by_id:
            matching_loser_poses = loser_by_id[protein_id]
            
            # Create pairs with each matching loser pose
            for loser_pose in matching_loser_poses:
                pair = {
                    "id": protein_id,
                    "winner_path": winner_pose['path'],
                    "loser_path": loser_pose['path'],
                    "winner_dockq": winner_pose['dockq'],
                    "loser_dockq": loser_pose['dockq'],
                    "winner_type": winner_pose['generation_method'],
                    "loser_type": loser_pose['generation_method']
                }
                preference_pairs.append(pair)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save preference pairs
    with open(output_path, 'w') as f:
        json.dump(preference_pairs, f, indent=2)
    
    print(f"Created {len(preference_pairs)} preference pairs")
    print(f"Saved to: {output_path}")
    
    return preference_pairs

def process_subdirectory_batch(gpu_id, subdirs_batch, receptor_ligand_mapping):
    """Process a batch of subdirectories on a specific GPU"""
    torch.cuda.set_device(gpu_id)
    local_converter = PDBToTorchConverter()
    
    for subdir in subdirs_batch:
        try:
            # Get receptor and ligand chain IDs
            if subdir not in receptor_ligand_mapping:
                print(f"Warning: {subdir} not found in receptor_ligand_mapping, skipping")
                continue
            
            chain_mapping = receptor_ligand_mapping[subdir]
            custom_chain_ids = {
                'receptor': chain_mapping['receptor'],
                'ligand': chain_mapping['ligand']
            }
            
            # Define paths
            subdir_path = f'/scratch/jgray21/rzhu41/684/capri_scoreset/{subdir}'
            pdb_folder = os.path.join(subdir_path, 'pdb')
            pt_folder = os.path.join(subdir_path, 'pt')
            
            # Create pt folder if it doesn't exist
            os.makedirs(pt_folder, exist_ok=True)
            
            # Find all PDB files in the pdb folder
            if not os.path.exists(pdb_folder):
                print(f"Warning: PDB folder {pdb_folder} does not exist, skipping {subdir}")
                continue
            
            pdb_files = glob.glob(os.path.join(pdb_folder, '*.pdb'))
            print(f"GPU {gpu_id}: Processing {len(pdb_files)} PDB files in {subdir}")
            
            for pdb_file in pdb_files:
                try:
                    # Get filename without extension
                    base_name = os.path.splitext(os.path.basename(pdb_file))[0]
                    pt_file = os.path.join(pt_folder, f'{base_name}.pt')
                    
                    # Skip if PT file already exists
                    if os.path.exists(pt_file):
                        continue
                    
                    # Convert PDB to PT
                    local_converter.convert_pdb_to_pt(
                        pdb_path=pdb_file,
                        output_path=pt_file,
                        esm_embeddings_provided=False,
                        custom_receptor_ligand_chain_id=custom_chain_ids
                    )
                    
                except Exception as e:
                    print(f"GPU {gpu_id}: Error processing {pdb_file}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"GPU {gpu_id}: Error processing subdirectory {subdir}: {str(e)}")
            continue

if __name__ == "__main__":

    EXPERIMENT = "data_vis" #"data_gen"

    if EXPERIMENT=="data_gen":
        main()
    elif EXPERIMENT == "convert_pdb_to_pt":
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        # Read the list of subdirectories
        with open('/scratch/jgray21/rzhu41/684/capri_score_set_dips_cleaned/nonoverlapping_with_dips_complexes.txt', 'r') as f:
            subdirectories = [line.strip() for line in f if line.strip()]
        
        # Load the receptor-ligand mapping
        with open('/scratch/jgray21/rzhu41/684/capri_score_set_dips_cleaned/receptor_ligand_mapping.json', 'r') as f:
            receptor_ligand_mapping = json.load(f)
        
        # Get number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs for parallel processing")
        
        # Split subdirectories across GPUs
        if num_gpus > 1:
            # Create batches for each GPU
            subdirs_per_gpu = len(subdirectories) // num_gpus
            gpu_batches = []
            
            for gpu_id in range(num_gpus):
                start_idx = gpu_id * subdirs_per_gpu
                if gpu_id == num_gpus - 1:  # Last GPU gets remaining subdirectories
                    end_idx = len(subdirectories)
                else:
                    end_idx = (gpu_id + 1) * subdirs_per_gpu
            
                gpu_batches.append(subdirectories[start_idx:end_idx])
            
            # Start multiprocessing
            processes = []
            for gpu_id, batch in enumerate(gpu_batches):
                if len(batch) > 0:
                    p = mp.Process(target=process_subdirectory_batch, args=(gpu_id, batch, receptor_ligand_mapping))
                    p.start()
                    processes.append(p)
            
            # Wait for all processes to complete
            for p in processes:
                p.join()
        else:
            # Single GPU processing
            process_subdirectory_batch(0, subdirectories)
        
        print("PDB to PT conversion completed!")
    elif EXPERIMENT == "data_vis":
        SLURM_JOB_IDS = ["307336", "307337", "307338", "307339", "307745", "307750", "307753"]  # Example SLURM job IDs
        POSE_RANKING_DATASET = "db5_all"
        MAX_TRANSLATION = "varied!"
        MAX_ROTATION = "varied!"
        NUM_PROT_TO_VISUALIZE = 8  # Number of proteins to visualize
        output_dir = f"/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO"
        output_filename = "systematic_dockQ_distribution_vis.png"

        import matplotlib.pyplot as plt

        # Read all poses from all files
        all_poses_data = []
        for SLURM_JOB_ID in SLURM_JOB_IDS:
            individual_poses_list = [
                f"/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/{SLURM_JOB_ID}_dips_hetero_model_0/run0/rank_0/individual_poses.json",
                f"/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/{SLURM_JOB_ID}_dips_hetero_model_0/run0/rank_1/individual_poses.json",
                f"/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/{SLURM_JOB_ID}_dips_hetero_model_0/run0/rank_2/individual_poses.json",
                f"/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/{SLURM_JOB_ID}_dips_hetero_model_0/run0/rank_3/individual_poses.json"
            ]
            for poses_path in individual_poses_list:
                with open(poses_path, 'r') as f:
                    poses = json.load(f)
                all_poses_data.extend(poses)
        
        # Group poses by protein ID
        poses_by_id = defaultdict(list)
        for pose in all_poses_data:
            poses_by_id[pose['id']].append(pose['dockq'])
        
        print(f"Total poses: {len(all_poses_data)}")
        print(f"Unique protein IDs: {len(poses_by_id)}")
        
        # Create visualization
        plt.figure(figsize=(30, 10))
        
        # Create violin plot to show distribution for each protein
        X = 95  # User-defined percentile
        protein_ids = sorted(poses_by_id.keys(), key=lambda pid: np.percentile(poses_by_id[pid], X))
        step = max(len(protein_ids) // NUM_PROT_TO_VISUALIZE, 1)
        protein_ids = [protein_ids[i] for i in range(0, len(protein_ids), step)][:NUM_PROT_TO_VISUALIZE]
        output_filename = f"systematic_dockQ_distribution_vis_{X}percentile.png"
        dockq_data = [poses_by_id[pid] for pid in protein_ids]
        
        # Use violin plot for distribution visualization
        parts = plt.violinplot(dockq_data, positions=range(len(protein_ids)), 
                  widths=0.7, showmeans=True)
        
        # Customize colors for better visibility
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        # Overlay scatter plot of actual points with jitter
        for i, dockq_values in enumerate(dockq_data):
            # Add random jitter to x positions
            # Scale jitter based on number of plots (optimized for 15 plots)
            jitter_scale = min(4.5 / len(protein_ids), 0.1)
            jitter = np.random.uniform(-jitter_scale, jitter_scale, len(dockq_values))
            x_positions = [i + j for j in jitter]
            plt.scatter(x_positions, dockq_values, s=8, alpha=0.6, color='darkblue')
        
        # Add reference lines for key DockQ thresholds
        plt.axhline(y=0.23, color='orange', linestyle='--', alpha=0.7, label='DockQ = 0.23 (Acceptable)')
        plt.axhline(y=0.49, color='red', linestyle='--', alpha=0.7, label='DockQ = 0.49 (Medium)')
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='DockQ = 0.8 (High)')
        
        # Customize plot
        plt.xlabel('Protein ID Index')
        plt.ylabel('DockQ Score')
        plt.title(f'DockQ Score Distribution for Perturbed GT ({MAX_TRANSLATION}A tr., {MAX_ROTATION}deg. rot.)\n(Violin Plot)')
        plt.legend(fontsize=30, loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Set y-axis limits
        plt.ylim(-0.1, 1.1)
        plt.yticks(fontsize=30)
        
        # If too many proteins, adjust x-axis
        if len(protein_ids) > 50:
            plt.xticks([])  # Remove x-tick labels if too many
            plt.xlabel('Protein ID Index (labels removed for clarity)')
        else:
            plt.xticks(range(len(protein_ids)), [pid[:8] for pid in protein_ids], rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")

    elif EXPERIMENT == 1:
        # Experiment 1: Ground Truth vs Inference with DockQ > 0.23
        winner_poses_list = ["/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295646_dips_hetero_model_0/run0/rank_0/individual_poses.json"]
        loser_poses_list = ["/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/294381_dips_hetero_model_0/run0/individual_poses.json"]
        output_filename = "preference_pairs_gt_vs_above0.23inference.json"
        
        create_preference_pairs(
            winner_poses_list=winner_poses_list,
            loser_poses_list=loser_poses_list,
            output_filename=output_filename,
            winner_methods=['ground_truth'],
            loser_methods=['inference'],
            loser_min_threshold=0.23
        )
    elif EXPERIMENT == 2:
        # Experiment 2: Ground Truth vs All Inference Poses
        winner_poses_list = ["/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295646_dips_hetero_model_0/run0/rank_0/individual_poses.json"]
        loser_poses_list = [
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/294381_dips_hetero_model_0/run0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_1/individual_poses.json"
        ]
        output_filename = "preference_pairs_gt_vs_all_inference.json"

        create_preference_pairs(
            winner_poses_list=winner_poses_list,
            loser_poses_list=loser_poses_list,
            output_filename=output_filename,
            winner_methods=['ground_truth'],
            loser_methods=['inference']
        )
    elif EXPERIMENT == 3:
        # Experiment 3: Ground Truth + Inference Poses above 0.49 vs Inference Poses below 0.49 but above 0.23
        winner_poses_list = [
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295646_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/294381_dips_hetero_model_0/run0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_1/individual_poses.json"
        ]
        loser_poses_list = [
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/294381_dips_hetero_model_0/run0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_1/individual_poses.json"
        ]
        output_filename = "preference_pairs_gt_infAbove0.49_vs_infBetween0.23And0.49.json"

        create_preference_pairs(
            winner_poses_list=winner_poses_list,
            loser_poses_list=loser_poses_list,
            output_filename=output_filename,
            winner_methods=['ground_truth', 'inference'],
            winner_min_threshold=0.49,
            loser_methods=['inference'],
            loser_min_threshold=0.23,
            loser_max_threshold=0.49
        )
    elif EXPERIMENT == 4:
        # Experiment 4: Ground Truth + Inference Poses above 0.49 vs Inference Poses below 0.49
        winner_poses_list = [
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295646_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/294381_dips_hetero_model_0/run0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_1/individual_poses.json"
        ]
        loser_poses_list = [
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/294381_dips_hetero_model_0/run0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_1/individual_poses.json"
        ]
        output_filename = "preference_pairs_gt_infAbove0.49_vs_infBelow0.49.json"

        create_preference_pairs(
            winner_poses_list=winner_poses_list,
            loser_poses_list=loser_poses_list,
            output_filename=output_filename,
            winner_methods=['ground_truth', 'inference'],
            winner_min_threshold=0.49,
            loser_methods=['inference'],
            loser_max_threshold=0.49
        )
    elif EXPERIMENT == 5:
        # Experiment 5: Inference Poses above 0.49 vs Inference Poses below 0.49 and above 0.23
        winner_poses_list = [
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295646_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/294381_dips_hetero_model_0/run0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_1/individual_poses.json"
        ]
        loser_poses_list = [
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/294381_dips_hetero_model_0/run0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_1/individual_poses.json"
        ]
        output_filename = "preference_pairs_infAbove0.49_vs_infBetween0.23And0.49.json"

        create_preference_pairs(
            winner_poses_list=winner_poses_list,
            loser_poses_list=loser_poses_list,
            output_filename=output_filename,
            winner_methods=['inference'],
            winner_min_threshold=0.49,
            loser_methods=['inference'],
            loser_max_threshold=0.49,
            loser_min_threshold=0.23
        )
    elif EXPERIMENT == 6:
        # Experiment 6: Inference Poses above 0.49 vs Inference Poses below 0.49
        winner_poses_list = [
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295646_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/294381_dips_hetero_model_0/run0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_1/individual_poses.json"
        ]
        loser_poses_list = [
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/294381_dips_hetero_model_0/run0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_1/individual_poses.json"
        ]
        output_filename = "preference_pairs_infAbove0.49_vs_infBelow0.49.json"

        create_preference_pairs(
            winner_poses_list=winner_poses_list,
            loser_poses_list=loser_poses_list,
            output_filename=output_filename,
            winner_methods=['inference'],
            winner_min_threshold=0.49,
            loser_methods=['inference'],
            loser_max_threshold=0.49
        )
    elif EXPERIMENT == "energy_finetune":
        all_poses = [
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295646_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/294381_dips_hetero_model_0/run0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295734_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295735_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295736_dips_hetero_model_0/run0/rank_1/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_0/individual_poses.json",
            "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295737_dips_hetero_model_0/run0/rank_1/individual_poses.json"
        ]

        # Read all poses from all files
        all_poses_data = []
        for poses_path in all_poses:
            with open(poses_path, 'r') as f:
                poses = json.load(f)
                all_poses_data.extend(poses)
        
        # Filter to only keep 'inference' and 'ground_truth' poses
        filtered_poses = [
            pose for pose in all_poses_data 
            if pose['generation_method'] in ['inference', 'ground_truth']
        ]
        
        print(f"Total poses after filtering: {len(filtered_poses)}")
        
        # Group by ID and create the desired structure
        grouped_data = {}
        
        for pose in filtered_poses:
            protein_id = pose['id']
            
            if protein_id not in grouped_data:
                grouped_data[protein_id] = {
                    'id': protein_id,
                    'training_pose': '',
                    'gt_pose': '',
                    'medium_poses': [],
                    'acceptable_poses': [],
                    'bad_poses': []
                }
            
            if pose['generation_method'] == 'ground_truth':
                # Confirm only one GT pose per ID
                if grouped_data[protein_id]['gt_pose'] != '':
                    print(f"Warning: Multiple ground truth poses found for {protein_id}")
                grouped_data[protein_id]['gt_pose'] = pose['path']
            
            elif pose['generation_method'] == 'inference':
                dockq = pose['dockq']
                if dockq > 0.49:
                    grouped_data[protein_id]['medium_poses'].append(pose['path'])
                elif 0.23 < dockq <= 0.49:
                    grouped_data[protein_id]['acceptable_poses'].append(pose['path'])
                else:
                    grouped_data[protein_id]['bad_poses'].append(pose['path'])
        
        # Set training_pose path for each protein ID
        for protein_id in grouped_data.keys():
            training_pose_path = f"/scratch/jgray21/rzhu41/DFMDock2/src/data/pt/dips_bb/{protein_id}.pt"
            grouped_data[protein_id]['training_pose'] = training_pose_path
        
        # Convert to list
        final_data = list(grouped_data.values())
        
        print(f"Number of unique protein IDs: {len(final_data)}")
        
        # Calculate statistics
        medium_lengths = [len(entry['medium_poses']) for entry in final_data]
        acceptable_lengths = [len(entry['acceptable_poses']) for entry in final_data]
        bad_lengths = [len(entry['bad_poses']) for entry in final_data]
        
        print(f"Medium poses - Average: {np.mean(medium_lengths):.2f}, Std: {np.std(medium_lengths):.2f}, Min: {np.min(medium_lengths)}")
        print(f"Acceptable poses - Average: {np.mean(acceptable_lengths):.2f}, Std: {np.std(acceptable_lengths):.2f}, Min: {np.min(acceptable_lengths)}")
        print(f"Bad poses - Average: {np.mean(bad_lengths):.2f}, Std: {np.std(bad_lengths):.2f}, Min: {np.min(bad_lengths)}")

        # Save the final data to the specified path
        output_dir = "/scratch/jgray21/rzhu41/DFMDock2/data/energy_finetune"
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "july14_dataset_0.json")

        with open(output_file, 'w') as f:
            json.dump(final_data, f, indent=2)

        print(f"Saved {len(final_data)} protein entries to {output_file}")
        
        