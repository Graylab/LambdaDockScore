#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=a100,h100
#SBATCH --gres=gpu:1
#SBATCH --account=[insert your SLURM account]
#SBATCH --time=48:00:00
#SBATCH --error=slogs/db55_sample_and_score_%j.err
#SBATCH --output=slogs/db55_sample_and_score_%j.out

# Change to the directory this script is in
# DFMDock source checkout with the sampler (override to relocate).
BASE_DIR="${DFMDOCK_SRC:-../DFMDock}"
cd "${BASE_DIR}"

# Set default values using relative paths
CKPT_DIR="checkpoints/test_ckpts"  # original value: ../checkpoints
RANKING_MODEL="308593_epoch_36-step_99197"
TEST_SET="db5_all"
DIR="src/DFMDock_vanilla_ScoreNet_w_separate_ranking_model_${RANKING_MODEL}_${TEST_SET}_save_both_outputs"         # original value: $(pwd)
TRAIN_SET="dips_hetero"                                       # original value: "dips_bb"
MODEL="model_0"
NUM_SAMPLES=15                                                # original value: 120
NUM_STEPS=40
NOISE_SCALE=0.5
RUN=${1:-0}  # default to 0 if not provided

# Execute the code
python "src/inference_mlsb_Score_Net_save_both_outputs.py" \
  data.ckpt="${CKPT_DIR}/${TRAIN_SET}/${MODEL}.ckpt" \
  data.dataset="${TEST_SET}" \
  data.out_csv_dir="${DIR}/csv_files/" \
  data.out_csv="${TEST_SET}_${MODEL}_${NOISE_SCALE}_${NUM_SAMPLES}_samples_${NUM_STEPS}_steps_${TRAIN_SET}_${RUN}.csv" \
  data.num_samples="${NUM_SAMPLES}" \
  data.num_steps="${NUM_STEPS}" \
  data.out_pdb=True \
  data.out_pdb_dir="${DIR}/pdbs/${TEST_SET}_${MODEL}_${NOISE_SCALE}_${NUM_SAMPLES}_samples_${NUM_STEPS}_steps_${TRAIN_SET}/run${RUN}" \
  data.out_trj=True \
  data.out_trj_dir="${DIR}/trjs/${TEST_SET}_${MODEL}_${NOISE_SCALE}_${NUM_SAMPLES}_samples_${NUM_STEPS}_steps_${TRAIN_SET}/run${RUN}" \
  data.test_all=True \
  data.use_clash_force=False \
  data.tr_noise_scale="${NOISE_SCALE}" \
  data.rot_noise_scale="${NOISE_SCALE}" \
  data.use_finetuned_ranking_model=True \
  data.finetune_ranking_model_ckpt="${CKPT_DIR}/${TRAIN_SET}/${RANKING_MODEL}.ckpt" \
  # data.randomize=True