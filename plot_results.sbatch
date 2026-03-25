#!/bin/bash
#SBATCH --job-name=plot_rewl
#SBATCH --time=0-00:10:00 # Adjust time as needed
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G # Adjust memory as needed, usually not much
#SBATCH --output=plot_rewl_%j.out

# Navigate to your project directory
cd ~/rewl_Project/


# Run the plotting script
module load anaconda ### load anaconda module
source activate my_env ### activate a conda environment, replace my_env with your conda environment
python plot_results.py

# Optional: Copy plots to a specific location if not already saving directly in rewl_Project/
# For this setup, they are saved directly in ~/rewl_Project/plots/ so no extra cp is needed
