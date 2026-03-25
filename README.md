# Revisiting the Volume Hypothesis

This research was submitted to the **ICML conference**. This code is part of a Master's thesis research at **Ben-Gurion University**, supervised by **Dr. Ari Pakman** and **Dr. Yakir Berchenko**.

## 🚀 Project Overview
Modern deep neural networks often contain far more parameters than needed to fit their training data, yet they achieve impressive generalization. A common explanation for this success is the implicit bias of stochastic gradient descent (SGD). An alternative **Volume Hypothesis** posits that, within low training-loss regions, basins leading to strong generalization occupy much larger regions of weight space than basins that generalize poorly, and therefore SGD is simply more likely to land in the former.

Recent experimental explorations of this idea present seemingly contradictory results. This project explores these contradictions using the **Replica Exchange Wang–Landau (REWL)** algorithm to estimate the joint density of states over training and test accuracies in binary networks. Across several architectures and datasets, we show that the generalization advantage of SGD over random sampling diminishes as the training data size grows, suggesting a resolution to the paradox.

### Key Components:
* **`Walker` Class (`utils.py`)**: Manages local state, weight mutations (spin flips), and accuracy tracking for specific windows.
* **Replica Exchange (`exchange` function in `utils.py`)**: Enables walkers in adjacent accuracy windows to swap configurations to ensure global convergence.
* **BinaryConnect (`binaryconnect.py`)**: Implements weight binarization and clipping to maintain a binary-weight manifold during simulation.
* **Multi-GPU Support**: Configured for distributed execution using PyTorch's `gloo` backend.

---

## 💻 Running the Simulation

The primary focus of this project is execution within an **HPC Cluster** environment using **Apptainer** due to specific constraints where standard container orchestration tools are unavailable.

### HPC Cluster Execution
The provided `.sbatch` files automate the setup of temporary scratch directories, dataset management, and result synchronization.

* **Main Simulation (`submit_simulation.sbatch`)**:
    * Allocates a GPU node (e.g., `rtx6000`) and executes the simulation using `torch.distributed.run`.
    * Utilizes a local scratch directory to handle heavy I/O during the run and automatically syncs results back to the project folder.
* **Live Monitoring & Plotting (`plot_results.sbatch`)**:
    * Because simulations for large datasets can be very long, this script is designed to be executed **during the run**.
    * Running this on the fly allows you to generate and inspect plots from intermediate `.npz` results while the simulation is still active, providing real-time insights into the density of states estimation.

### Local Execution
While the focus is on cluster deployment, local execution is also possible using the provided `compose.yaml` for GPU-shared environments.

---

## 🛠 Configuration & Walker Variants

The repository includes several versions of the walker scripts to accommodate different hardware strategies and simulation scales:

* **Standard 6-GPU Run (`wang_landau_walkers.py`)**: Designed for a setup where each of the 6 walkers is assigned a dedicated GPU (e.g., `rank % 6`).
* **Parallel Simulations (`wang_landau_walkers_A.py` & `wang_landau_walkers_B.py`)**: Optimized for efficiency by allowing 2 walkers to share a single GPU (e.g., `rank % 3`). This enables running two independent simulations in parallel across 6 GPUs (12 walkers total).
* **Model Selection**: The `utils.py` file contains several architectures, including `SimpleCNN`, `SimpleCNN_deep`, and `SimpleCNN_wide`. You can specify which model to use by adjusting the `model_class` in `config.py`.

---

## 📊 Results & Visualization
The simulation outputs `.npz` files for each rank containing the histogram (`h`) and the estimated log density of states (`log_g`).

To process these files and stitch the accuracy windows together, run:
```bash
python plot_results.py
```
This will generate visualizations in the `./plots` directory.

---

## 📁 Repository Structure
* `wang_landau_walkers.py`: Main entry point for 6-GPU simulations.
* `wang_landau_walkers_A.py` / `_B.py`: Variants for parallel simulations with GPU sharing.
* `config.py`: Central configuration for accuracy bins, windows, and hyperparameters.
* `utils.py`: Contains the `Walker` logic and various CNN architectures.
* `submit_simulation.sbatch`: Slurm script for primary cluster execution.
* `plot_results.sbatch`: Slurm script for live monitoring and intermediate plotting.