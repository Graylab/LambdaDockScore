#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --account=[insert your SLURM account]
#SBATCH --time=10:00:00
#SBATCH --error=slogs/dpo_data_gen_%j.err
#SBATCH --output=slogs/dpo_data_gen_%j.out

# Repo root: defaults to two levels up from this script; override REPO_ROOT to relocate.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slogs

# Set default values using relative paths
CKPT_DIR="checkpoints/test_ckpts"                            # Checkpoint directory
TRAIN_SET="dips_hetero"                                      # Training set used for model
MODEL="model_0"                                              # Model checkpoint name
RUN=${1:-0}                                                  # Run number (default to 0 if not provided)
PAIRS_PER_HIGH_DOCKQ_SAMPLE=5                                # Number of pairs per high DockQ sample

# The dataset we are generating poses from
POSE_DATASET="db5_all"                     # Dataset to sample from for generating poses
SAMPLE_POSE_SET=-1                                       # Number of samples to use from the pose set (if -1, then use all. if >-1, then we are just generating data for a subset for debugging)
POSE_SET_SUBSET_INDICES='[168,253]' #[0,2681], '[2681,5362]', [5362,8043], [8043,10724]  #Break DIPS pose set into four segments, [-1,-1] for smaller datasets that don't need subsetting

# Output directory for generated data (override DATA_ROOT to relocate)
DATA_ROOT="${DATA_ROOT:-data/DockDPO}"
OUTPUT_DIR="${DATA_ROOT}/${SLURM_JOB_ID}_${TRAIN_SET}_${MODEL}_${POSE_DATASET}/run${RUN}"  # generated decoy poses

# Perturbation parameters
MAX_TRANSLATION='[0.1,1.0,2.5]'                                          # Max translation for GT perturbation
MAX_ROTATION='[1.0,7.0,20.0]'                                            # Max rotation (degrees) for GT perturbation
NUM_PERTURBATION_SAMPLES='[20,100,100]'                               # Number of perturbation samples to generate per ground truth sample
PERTURBATION_TARGET_DOCKQ=0                                   # Target DockQ threshold for GT perturbations
MAX_PERTURBATION_ATTEMPTS=1000                               # Max attempts to generate valid GT perturbations
ROT_TYPE="normal"                                          # Rotation type for perturbation ('uniform' or 'normal')
TR_TYPE="normal"                                           # Translation type for perturbation ('uniform' or 'normal')

# Inference parameters
INFERENCE_TARGET_DOCKQ=0                                    # Target DockQ for inference
MAX_INFERENCE_ATTEMPTS=1000                                    # Max attempts to generate valid inference samples
NUM_INFERENCE_SAMPLES=50                                   # Number of inference samples to generate

# Decide what to generate
GENERATE_PAIRS=False                                       # Whether to generate pairs
GENERATE_GROUND_TRUTH=False                                # Whether to generate ground truth samples
GENERATE_INFERENCE=True                                   # Whether to generate inference samples
GENERATE_PERTURBED_GT=True                                # Whether to generate perturbed ground truth samples

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Print job information
echo "Starting DPO data generation job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Model checkpoint: ${CKPT_DIR}/${TRAIN_SET}/${MODEL}.ckpt"
echo "Output directory: $OUTPUT_DIR"
echo "Number of samples per type: $NUM_SAMPLES"
echo "Target DockQ threshold: $TARGET_DOCKQ"
echo "Run number: $RUN"
echo "----------------------------------------"

# Execute the DPO data generation pipeline
python "src/DockDPO_data_gen.py" \
  data.ckpt="${CKPT_DIR}/${TRAIN_SET}/${MODEL}.ckpt" \
  data.output_dir="${OUTPUT_DIR}" \
  data.dataset="${POSE_DATASET}" \
  data.num_perturbation_samples="${NUM_PERTURBATION_SAMPLES}" \
  data.perturbation_target_dockq="${PERTURBATION_TARGET_DOCKQ}" \
  data.max_translation="${MAX_TRANSLATION}" \
  data.max_rotation="${MAX_ROTATION}" \
  data.tr_type="${TR_TYPE}" \
  data.rot_type="${ROT_TYPE}" \
  data.sample_pose_set="${SAMPLE_POSE_SET}" \
  data.pose_set_subset_indices="${POSE_SET_SUBSET_INDICES}" \
  data.max_perturbation_attempts="${MAX_PERTURBATION_ATTEMPTS}" \
  data.pairs_per_high_dockq_sample="${PAIRS_PER_HIGH_DOCKQ_SAMPLE}" \
  data.use_esm=True \
  data.generate_pairs=${GENERATE_PAIRS} \
  data.generate_ground_truth=${GENERATE_GROUND_TRUTH} \
  data.generate_inference=${GENERATE_INFERENCE} \
  data.generate_perturbed_gt=${GENERATE_PERTURBED_GT} \
  data.inference_target_dockq="${INFERENCE_TARGET_DOCKQ}" \
  data.max_inference_attempts="${MAX_INFERENCE_ATTEMPTS}" \
  data.num_inference_samples="${NUM_INFERENCE_SAMPLES}" \