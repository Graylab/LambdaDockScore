import sys
import os
import torch
import warnings

# Add src to path if running from root
sys.path.append(os.path.join(os.path.dirname(__file__)))

from Ranking_Net import Ranking_Model
from datasets.ppi_mlsb_dataset import PPIDataset

def score_single_pt(pt_file, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    print(f"Loading model from {checkpoint_path}...")
    try:
        model = Ranking_Model.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
        )
    except Exception as e:
        print(f"Failed to load as Ranking_Model: {e}")
        print("Trying to load as Score_Model...")
        from models.score_model_mlsb import Score_Model
        model = Score_Model.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
        )
    
    model.eval()
    model.to(device)

    # Load the .pt file
    print(f"Loading data from {pt_file}...")
    raw_data = torch.load(pt_file, map_location='cpu', weights_only=False)
    _id = os.path.basename(pt_file).replace('.pt', '')

    # Use PPIDataset helper to construct the batch
    # We create a dummy dataset instance to access the method
    dataset_helper = PPIDataset(dataset=None, training=False)
    
    # Construct the batch
    # Note: construct_full_output expects raw_data to be a HeteroData object or similar dict
    batch = dataset_helper.construct_full_output(raw_data=raw_data, _id=_id, swap=False)

    # Move batch to device and add batch dimension if necessary
    # The model expects batched inputs. construct_full_output returns unbatched tensors (N, D).
    # We need to wrap them in a batch or add a dimension if the model expects it.
    # However, inference_mlsb_Ranking_Net.py passes the output of construct_full_output (via dataloader)
    # directly to the model. The dataloader adds a batch dimension (1, N, D).
    # But inference_mlsb_Ranking_Net.py squeezes it: batch['rec_x'].squeeze(0).
    # So the model expects (N, D).
    
    batch_input = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_input[k] = v.to(device)
        else:
            batch_input[k] = v
            
    # Add dummy time 't' as required by Score_Model/Ranking_Model
    batch_input["t"] = torch.zeros(1, device=device) + 1e-5

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # Ranking_Model.get_energy expects a batch dict.
        # It calls self.net(batch, return_energy=True).
        # Let's try calling the model directly as in inference_mlsb_Ranking_Net.py
        output = model(batch_input)
    
    print("Results:")
    print(f"Energy: {output['energy'].item()}")
    if 'num_clashes' in output:
        print(f"Num Clashes: {output['num_clashes'].item()}")

if __name__ == "__main__":
    pt_file = "/scratch/jgray21/rzhu41/LambdaDockScore/data/example_T025.1_T25_U01.M01.pt"
    checkpoint_path = "/scratch/jgray21/rzhu41/LambdaDockScore/checkpoints/best_checkpoint.ckpt"
    
    if not os.path.exists(pt_file):
        print(f"Error: File {pt_file} not found.")
    elif not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
    else:
        score_single_pt(pt_file, checkpoint_path)
