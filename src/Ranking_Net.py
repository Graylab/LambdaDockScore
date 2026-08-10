import os
import copy
import torch
torch.set_float32_matmul_precision('medium')
import hydra
import warnings
import random
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import esm
from collections import defaultdict
from tqdm import tqdm
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from torch_geometric.data import HeteroData
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from scipy.spatial.transform import Rotation 
from omegaconf import OmegaConf
import json
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from functools import partial

# --- Import necessary components from your project files ---
from models.score_model_mlsb import Score_Model
from utils.so3_diffuser import SO3Diffuser 
from utils.r3_diffuser import R3Diffuser
from utils.geometry import axis_angle_to_matrix
from utils import residue_constants
from datasets.ppi_mlsb_dataset import relpos, get_interface_residue_tensors, PPIDataset
from DockDPO_data_gen import PDBToTorchConverter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#----------------------------------------------------------------------------
# Ranking Net Dataset
# This dataset loads a regular PPIDataset batch but also a list of poses with varying DockQ and then sampling K of those from a multinomial based on 10 even buckets between 0 and 1.

def load_embedding_from_reference(ref): #This is needed to load the embeddings for the edited .pt files
    """Load embedding from reference object"""
    
    data = torch.load(ref, weights_only=False)
    rec_embedding = data['receptor'].x.squeeze(0)
    lig_embedding = data['ligand'].x.squeeze(0)

    return rec_embedding, lig_embedding

class Ranking_Dataset(PPIDataset):
    def __init__(self, data_list_path, crop_size, use_esm=True, buckets=10, num_ranking_poses=10):
        """
        Args:
            data_list_path (str): Path to a CSV file with columns: 
                                  'id', 'winner_path', 'loser_path'.
            use_esm (bool): Whether to compute and use ESM embeddings.
        """
        super().__init__(dataset=None, use_esm=use_esm, crop_size=crop_size)
        # load JSON list with 'winner_path','loser_path','score_w','score_l'
        with open(data_list_path, 'r') as f:
            self.data = json.load(f)
        
        self.buckets = buckets
        self.num_ranking_poses = num_ranking_poses

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        _id = self.data[idx]['id']

        #First construct the original training sample
        if self.training:
            # Set a random boolean indicator for swapping receptor and ligand
            swap = random.random() < 0.5
        else:
            swap = False
        training_pose_raw = torch.load(self.data[idx]['training_pose'], weights_only=False)
        training_pose = self.construct_full_output(raw_data=training_pose_raw, _id=_id, swap=swap)

        #Now sample and construct all other poses
        ranking_poses_data = self.data[idx]['ranking_poses']
        
        # Split interval [0,1] into buckets
        bucket_edges = np.linspace(0, 1, self.buckets + 1)
        bucket_counts = np.zeros(self.buckets)
        pose_to_bucket = {}
        
        # Assign each pose to a bucket and count
        for i, (dockq_score, file_path) in enumerate(ranking_poses_data):
            bucket_idx = min(np.digitize(dockq_score, bucket_edges) - 1, self.buckets - 1)
            bucket_idx = max(bucket_idx, 0)  # Ensure non-negative
            pose_to_bucket[i] = bucket_idx
            bucket_counts[bucket_idx] += 1
        
        # Calculate probabilities: 1 / (number of poses in bucket)
        probabilities = np.zeros(len(ranking_poses_data))
        for i, bucket_idx in pose_to_bucket.items():
            if bucket_counts[bucket_idx] > 0:
                probabilities[i] = 1.0 / bucket_counts[bucket_idx]
        
        # Normalize probabilities
        probabilities = probabilities / probabilities.sum()
        
        # Sample without replacement using multinomial
        num_samples = min(self.num_ranking_poses, len(ranking_poses_data))
        sampled_indices = np.random.choice(
            len(ranking_poses_data), 
            size=num_samples, 
            replace=False, 
            p=probabilities
        )
        
        # Load the sampled poses
        ranking_poses = []
        ranking_poses_dockq_scores = []
        for idx_sample in sampled_indices:
            dockq_score, file_path = ranking_poses_data[idx_sample]
            if self.training:
                swap = random.random() < 0.5
            else:
                swap = False
            pose_raw = torch.load(file_path, weights_only=False)
            #load in embeddings from original source
            if hasattr(pose_raw, 'embedding_reference_info'):
                rec_embedding, lig_embedding = load_embedding_from_reference(pose_raw.embedding_reference_info)
                pose_raw['receptor'].x = rec_embedding
                pose_raw['ligand'].x = lig_embedding
            ranking_poses.append(self.construct_full_output(raw_data=pose_raw, _id=_id, swap=swap))
            ranking_poses_dockq_scores.append(dockq_score)

        return {
            'id':            _id,
            'training_pose': training_pose,
            'ranking_poses': ranking_poses,
            'ranking_poses_dockq_scores': ranking_poses_dockq_scores #ordered in the same way as ranking_poses
        }

#----------------------------------------------------------------------------
# Ranking Fine-tuning Lightning Module

class Ranking_Model(Score_Model):
    def __init__(self, ranking_conf: DictConfig, model: DictConfig, diffuser: DictConfig, experiment: DictConfig):
        """
        Initializes the Ranking Model trainer.

        Args:
            ranking_conf (DictConfig): Configuration for the Ranking Model training (lr, beta).
            diffuser (DictConfig): Configuration for the R3 and SO3 diffusers.
            model (DictConfig): Configuration for the model architecture.
            experiment (DictConfig): Configuration for the experiment settings.
        """
        # 1. Load the model, diffuser, and experiment configurations using parent class. This will also load in self.net, which is what we are trying to fine-tune. pl.LightningModule takes care of self.device, self.log, and we inherit test_step() from Score_Model.
        super().__init__(
          model=model,
          diffuser=diffuser,
          experiment=experiment
        )
        self.save_hyperparameters("ranking_conf") #other config parameters are saved in the parent class

        # 2. Declare ranking-specific parameters
        self.ranking_conf = ranking_conf
        self.ranking_lr = ranking_conf.lr
        self.ranking_weight_decay = ranking_conf.weight_decay
        self.ranking_loss_weight = ranking_conf.ranking_loss_weight
        self.scaling_coeff_for_ranking_loss = ranking_conf.scaling_coeff_for_ranking_loss

    def get_energy(self, batch):
        rec_x = batch['rec_x'].squeeze(0)
        lig_x = batch['lig_x'].squeeze(0)
        rec_pos = batch['rec_pos'].squeeze(0)
        lig_pos = batch['lig_pos'].squeeze(0)
        position_matrix = batch['position_matrix'].squeeze(0)
        ires = batch['ires'].squeeze(0)

        # wrap to a batch
        batch = {
            "rec_x": rec_x,
            "lig_x": lig_x,
            "rec_pos": rec_pos,
            "lig_pos": lig_pos,
            "position_matrix": position_matrix,
            "ires": ires,
            "t": torch.zeros(1, device=self.device),  # dummy timestep because we're just doing energy calculation
        }

        energy = self.net(batch, return_energy=True)

        return energy
    
    def ranking_step(self, batch, training_loss):
        ranking_poses = batch['ranking_poses']

        psi = torch.as_tensor(batch['ranking_poses_dockq_scores'], device=self.device, dtype=torch.float32)  # (K,)

        energies = torch.stack([self.get_energy(p) for p in ranking_poses], dim=0)  # (K,)

        s = -1 * energies  # (K,)  # we want to minimize energy, so highest score goes to lowest energy

        ranking_loss = torch.tensor(0.0, device=self.device)

        K = energies.size(0)
        if K < 2:
            ranking_loss = torch.tensor(0.0, device=self.device)
        else:
            # compute dynamic ranks τ(i) induced by scores (highest score (lowest energy) → rank 1)
            # ranks will be floats 1,…,K
            _, sorted_idx = torch.sort(s, descending=True)
            ranks = torch.empty_like(s, device=self.device, dtype=torch.float32)
            ranks[sorted_idx] = torch.arange(1, K+1, device=self.device, dtype=torch.float32)

            # 5) compute gain G_i = 2^ψ_i − 1
            G = (2.0 ** psi) - 1.0  # (K,)

            # 6) build pairwise matrices
            s_i = s.unsqueeze(1)           # (K,1)
            s_j = s.unsqueeze(0)           # (1,K)
            G_i = G.unsqueeze(1)           # (K,1)
            G_j = G.unsqueeze(0)           # (1,K)
            rank_i = ranks.unsqueeze(1)    # (K,1)
            rank_j = ranks.unsqueeze(0)    # (1,K)

            # 7) λ weight: Δ_{i,j} = |G_i − G_j| · |1/D(rank_i) − 1/D(rank_j)|
            #    with D(k) = log2(1 + k)
            discount_i = 1.0 / torch.log2(1 + rank_i)
            discount_j = 1.0 / torch.log2(1 + rank_j)
            delta_gain = torch.abs(G_i - G_j)            # (K,K)
            delta_discount = torch.abs(discount_i - discount_j)  # (K,K)
            lambda_w = delta_gain * delta_discount      # (K,K)

            # 8) pairwise logistic loss ℓ_{i,j} = log2(1 + exp(−(s_i − s_j)))
            pair_loss = torch.log2(1 + torch.exp(-(s_i - s_j)))  # (K,K)

            # 9) only sum over pairs where ψ_i > ψ_j
            preference_mask = (psi.unsqueeze(1) > psi.unsqueeze(0)).float()  # (K,K)

            # 10) aggregate
            total_pairs = preference_mask.sum()
            if total_pairs > 0:
                ranking_loss = (lambda_w * pair_loss * preference_mask).sum() / total_pairs
            else:
                ranking_loss = torch.tensor(0.0, device=self.device)

        # 11) combine with your GT loss
        alpha = self.ranking_loss_weight
        total_loss = (1.0 - alpha) * training_loss + alpha * self.scaling_coeff_for_ranking_loss * ranking_loss

        return ranking_loss, total_loss

    def training_step(self, batch, batch_idx):
        """
        Batch will contain 1 training pose, and a list of poses for tuning the ranking loss.
        Following: https://arxiv.org/pdf/2402.01878 and https://storage.googleapis.com/gweb-research2023-media/pubtools/4591.pdf
        """
        training_loss = super().training_step(batch['training_pose'], batch_idx)

        ranking_loss, total_loss = self.ranking_step(batch, training_loss)

        self.log('train/ranking_loss', ranking_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Batch will contain 1 training pose, and a list of poses for tuning the ranking loss.
        Following: https://arxiv.org/pdf/2402.01878 and https://storage.googleapis.com/gweb-research2023-media/pubtools/4591.pdf
        """
        validation_loss = super().validation_step(batch['training_pose'], batch_idx)

        ranking_loss, total_loss = self.ranking_step(batch, validation_loss)

        self.log('val/ranking_loss', ranking_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        #Must overwrite this method to configure the optimizer for DPO fine-tuning
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.ranking_lr, 
            weight_decay=self.ranking_weight_decay
        )
        return optimizer

#----------------------------------------------------------------------------
# Ranking model fine-tuning execution block with Hydra for configuration management

@hydra.main(version_base=None, config_path="../configs", config_name="ranking_finetune")
def ranking_finetune(cfg: DictConfig):
    """
    Main function to run the ranking fine-tuning process.
    """
    print("---- FULL CONFIG ----")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    print("Starting ranking fine-tuning for DFMDock...")
    pl.seed_everything(cfg.seed)

    # Initialize Wandb logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        log_model=True
    )

    # --- Setup Dataset and DataLoader ---
    # This should point to a CSV file with columns: id, winner_path, loser_path, winner_dockq, loser_dockq
    ranking_train_set = Ranking_Dataset(data_list_path=cfg.data.ranking_train_set, crop_size=cfg.ranking_conf.crop_size, use_esm=cfg.ranking_conf.use_esm)
    ranking_train_dataloader = DataLoader(
        ranking_train_set, 
        batch_size=1, 
        shuffle=True, 
        num_workers=cfg.ranking_conf.num_workers
    )

    ranking_val_set = Ranking_Dataset(data_list_path=cfg.data.ranking_val_set, crop_size=cfg.ranking_conf.crop_size, use_esm=cfg.ranking_conf.use_esm)
    ranking_val_dataloader = DataLoader(
        ranking_val_set, 
        batch_size=1, 
        shuffle=False,
        num_workers=cfg.ranking_conf.num_workers
    )

    # --- Setup Ranking Lightning Module ---
    ranking_model = Ranking_Model.load_from_checkpoint(
        cfg.ranking_conf.ckpt_path, # Path to the pre-trained Ranking Model checkpoint
        ranking_conf=cfg.ranking_conf, # Supply just in case we load a checkpoint with no Ranking config
        map_location='cuda' if torch.cuda.is_available() else 'cpu',
        strict=False  # Allow loading with extra keys
    )

    # --- Setup PyTorch Lightning Trainer ---
    trainer = pl.Trainer(
        strategy=cfg.trainer.strategy,
        accelerator='auto',
        devices=cfg.trainer.devices,
        max_epochs=cfg.ranking_conf.max_epochs,
        log_every_n_steps=1,
        val_check_interval=cfg.ranking_conf.val_check_interval,
        logger=wandb_logger,
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath=cfg.ranking_conf.output_dir,
            save_top_k=-1,  # Save all checkpoints
            every_n_epochs=1,  # Save every epoch
            save_on_train_epoch_end=True,  # Save at epoch end
            monitor='val/total_loss', 
            mode='min'
        )]
    )

    # --- Start Training ---
    print("Beginning Ranking Model training...")
    trainer.fit(model=ranking_model, train_dataloaders=ranking_train_dataloader, 
                val_dataloaders=ranking_val_dataloader)
    print("Ranking Model fine-tuning finished.")

    # --- Save the fine-tuned model ---
    output_dir = cfg.ranking_conf.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'last_ckpt.ckpt')
    trainer.save_checkpoint(save_path)
    print(f"Fine-tuned model saved to: {save_path}")

def construct_ranking_dataset(list_of_pose_dirs, output_file_name):
    all_poses_data = []
        
    # Iterate through each directory in the list
    for pose_dir in list_of_pose_dirs:
        # Walk through the directory recursively
        for root, dirs, files in os.walk(pose_dir):
            # Look for 'individual_poses.json' files
            if 'individual_poses.json' in files:
                json_file_path = os.path.join(root, 'individual_poses.json')
                
                try:
                    # Read and parse the JSON file
                    with open(json_file_path, 'r') as f:
                        poses_data = json.load(f)
                    
                    # Ensure it's a list and extend our main list
                    if isinstance(poses_data, list):
                        all_poses_data.extend(poses_data)
                    else:
                        print(f"Warning: {json_file_path} does not contain a list of poses.")
                        
                except Exception as e:
                    print(f"Error reading {json_file_path}: {e}")
    
    print(f"Total poses collected: {len(all_poses_data)}")

    # Group by ID and create the desired structure
    grouped_data = {}

    for pose in all_poses_data:
        protein_id = pose['id']
        
        if protein_id not in grouped_data:
            grouped_data[protein_id] = {
                'id': protein_id,
                'training_pose': '',
                'ranking_poses': [],
            }
        
        # Append (dockq_score, file_path) to the ranking_poses list
        dockq_score = pose['dockq']
        file_path = pose['path']
        grouped_data[protein_id]['ranking_poses'].append((dockq_score, file_path))

    # Set training_pose path for each protein ID
    for protein_id in grouped_data.keys():
        training_pose_path = f"/scratch/jgray21/rzhu41/DFMDock2/src/data/pt/dips_bb/{protein_id}.pt" #location of both training and validation DIPS poses
        grouped_data[protein_id]['training_pose'] = training_pose_path
    
    # Convert to list
    final_data = list(grouped_data.values())
    
    print(f"Number of unique protein IDs: {len(final_data)}")

    # Save the final data to the specified path
    output_dir = "/scratch/jgray21/rzhu41/DFMDock2/data/ranking_finetune"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, output_file_name)

    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=4)

    print(f"Ranking dataset saved to: {output_file}")

if __name__ == "__main__":
    # For fine tuning
    ranking_finetune()

    # # For constructing the ranking train dataset
    # list_of_train_pose_dirs = [
    #     "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/295646_dips_hetero_model_0", #ground truth training poses
    #     "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/307336_dips_hetero_model_0",
    #     "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/307337_dips_hetero_model_0",
    #     "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/307338_dips_hetero_model_0",
    #     "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/307339_dips_hetero_model_0",
    # ]
    # train_file_name = "ranking_train_set_july17.json"

    # construct_ranking_dataset(list_of_pose_dirs=list_of_train_pose_dirs, output_file_name=train_file_name)

    # # For constructing the ranking validation dataset
    # list_of_val_pose_dirs = [
    #     "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/308466_dips_hetero_model_0", #ground truth validation poses
    #     "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/307745_dips_hetero_model_0",
    #     "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/307750_dips_hetero_model_0",
    #     "/scratch/jgray21/rzhu41/DFMDock2/data/DockDPO/307753_dips_hetero_model_0",
    # ]
    # val_file_name = "ranking_val_set_july17.json"

    # construct_ranking_dataset(list_of_pose_dirs=list_of_val_pose_dirs, output_file_name=val_file_name)

