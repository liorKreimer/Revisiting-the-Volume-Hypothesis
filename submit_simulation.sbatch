#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos= name_of_the_owner
##SBATCH --nodelist=ise-6000-02  # <--- ADDED LINE: Forces job onto this specific node
#SBATCH --job-name=Sim_B_REWL
#SBATCH --time=7-00:00:00
##SBATCH --gpus=6
#SBATCH --gres=gpu:6
#SBATCH --nodes=1
##SBATCH --ntasks=6
#SBATCH --mem=30G
#SBATCH --output=rewl_B_%j.out
#SBATCH --tmp=50G
##SBATCH --cpus-per-gpu=6
module load cuda/11.8

# Paths
SIF=~/containers/rewl_dual_sim.sif
SCRATCH_DIR=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}

# --- Data Preparation ---
echo "--- Starting Sim B Setup ---"
mkdir -p "$SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR/results"

# Copy files from the submission directory (~/rewl_sim/) to scratch
cp ~/rewl_Project/*.py "$SCRATCH_DIR/"
cp -r ~/rewl_Project/initial_weights "$SCRATCH_DIR/"
cp -r ~/rewl_Project/results "$SCRATCH_DIR/"
cp ~/rewl_Project/wl_data_tensors.pt "$SCRATCH_DIR/"

# Set env variables
export WLDATA="$SCRATCH_DIR/wl_data_tensors.pt"
export WLINIT="$SCRATCH_DIR/initial_weights"
export WLRESULTS="$SCRATCH_DIR/results"

cd "$SCRATCH_DIR"

# --- Execution ---
apptainer exec --nv "$SIF" \
    python -m torch.distributed.run \
        --nproc_per_node=6 \
        --nnodes=1 \
        --master_addr=localhost \
        --master_port=29500 \
        wang_landau_walkers.py

# --- Cleanup ---
cp -r "$SCRATCH_DIR/results" ~/rewl_Project/
