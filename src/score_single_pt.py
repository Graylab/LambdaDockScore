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
    if device.type == "cpu" and torch.get_num_threads() > 8:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
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
    import argparse
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_pt = os.path.join(repo_root, "data", "example_T026.1_T26_U20.M01.pt")
    default_ckpt = os.path.join(repo_root, "checkpoints", "lambdadockscore.ckpt")

    parser = argparse.ArgumentParser(description="Score a single .pt complex pose using a trained checkpoint.")
    parser.add_argument("--pt_file", default=default_pt, help="Path to input .pt HeteroData pose file")
    parser.add_argument("--checkpoint", default=default_ckpt, help="Path to model checkpoint (.ckpt)")
    args = parser.parse_args()

    if not os.path.exists(args.pt_file):
        print(f"Error: File {args.pt_file} not found.")
        sys.exit(1)
    elif not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        sys.exit(1)
    else:
        score_single_pt(args.pt_file, args.checkpoint)
