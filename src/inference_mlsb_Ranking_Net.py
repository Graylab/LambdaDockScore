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
from Ranking_Net import Ranking_Model
from datasets.ppi_mlsb_dataset import PPIDataset
from utils.pdb import save_PDB, place_fourth_atom 
from utils.metrics import compute_metrics

#----------------------------------------------------------------------------
# Data class for pose

@dataclass
class pose():
    _id: str
    rec_seq: str
    lig_seq: str
    rec_pos: torch.FloatTensor
    lig_pos: torch.FloatTensor
    index: str = None

#----------------------------------------------------------------------------
# Helper functions

def get_full_coords(coords):
    #get full coords
    N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
    # Infer CB coordinates.
    b = CA - N
    c = C - CA
    a = b.cross(c, dim=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
    
    O = place_fourth_atom(torch.roll(N, -1, 0),
                                    CA, C,
                                    torch.tensor(1.231),
                                    torch.tensor(2.108),
                                    torch.tensor(-3.142))
    full_coords = torch.stack(
        [N, CA, C, O, CB], dim=1)
    
    return full_coords

#----------------------------------------------------------------------------
# Scorer

class Scorer:
    def __init__(
        self,
        conf: DictConfig,
    ):
        self.data_conf = conf.data

        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load models
        self.model = Ranking_Model.load_from_checkpoint(
            self.data_conf.ckpt, 
            map_location=self.device,
        )
        self.model.eval()
        self.model.to(self.device)
        
        # get testset
        testset = PPIDataset(
            dataset=self.data_conf.dataset, 
            training=False, 
            use_esm=self.data_conf.use_esm,
        )

        # load dataset
        if self.data_conf.test_all:
            self.test_dataloader = data.DataLoader(testset, batch_size=1, num_workers=6)
        else:
            # get subset
            subset_indices = [0]
            subset = data.Subset(testset, subset_indices)
            self.test_dataloader = data.DataLoader(subset, batch_size=1, num_workers=6)
    
    def get_metrics(self, pred, label):
        metrics = compute_metrics(pred, label)
        return metrics
    
    def run_scoring(self):
        metrics_list = []
        for batch in tqdm(self.test_dataloader):
            # get batch from testset loader
            _id = batch['id'][0]
            rec_seq = batch['rec_seq'][0]
            lig_seq = batch['lig_seq'][0]
            rec_x = batch['rec_x'].to(self.device).squeeze(0)
            lig_x = batch['lig_x'].to(self.device).squeeze(0)
            rec_pos = batch['rec_pos'].to(self.device).squeeze(0)
            lig_pos = batch['lig_pos'].to(self.device).squeeze(0)
            position_matrix = batch['position_matrix'].to(self.device).squeeze(0)
            ires = batch['ires'].to(self.device).squeeze(0)

            batch_input = {
                "rec_x": rec_x,
                "lig_x": lig_x,
                "rec_pos": rec_pos.clone().detach(),
                "lig_pos": lig_pos.clone().detach(),
                "position_matrix": position_matrix,
                "ires": ires,
                "t": torch.zeros(1, device=self.device) + 1e-5
            }

            # get ground truth pose
            label = pose(
                _id=_id,
                rec_seq=rec_seq,
                lig_seq=lig_seq,
                rec_pos=rec_pos,
                lig_pos=lig_pos
            )

            with torch.no_grad():
                # Use get_energy from Ranking_Model
                # energy = self.model.get_energy(batch_input)
                # We use self.model(batch) to get full output including num_clashes if needed
                output = self.model(batch_input)

            metrics = {'id': _id}
            metrics.update(self.get_metrics([rec_pos, lig_pos], [rec_pos, lig_pos]))
            metrics.update({'energy': output["energy"].item()})
            if "num_clashes" in output:
                metrics.update({'num_clashes': output["num_clashes"].item()})
            metrics_list.append(metrics)
            
            print(metrics)

        return metrics_list
     
#----------------------------------------------------------------------------
# Main
@hydra.main(version_base=None, config_path="../configs", config_name="inference") 
def main(config: DictConfig):
    # Print the entire configuration
    print(OmegaConf.to_yaml(config))

    torch.manual_seed(0)
    scorer = Scorer(config)

    output_dir = config.data.out_csv_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # set output directory
    output_filename =  os.path.join(output_dir, config.data.out_csv)

    with open(output_filename, "w", newline="") as csvfile:
        results = scorer.run_scoring()

        # Write header row to CSV file
        if results:
            header = list(results[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

            for row in results:
                writer.writerow(row)

if __name__ == "__main__":
    main()
