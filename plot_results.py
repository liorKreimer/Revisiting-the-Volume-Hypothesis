#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 07:01:24 2025

@author: aripakman
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from config import eq_limits  # Make sure eq_limits is accessible via config.py

matplotlib.use('Agg')  # Set the backend to 'Agg' for non-interactive plotting


def load_walker_data(rank):
    """
    Helper function to load data from rank-specific .npz files.
    """
    samples_filename = f'./results/rank_{rank}_results.npz'
    try:
        res = np.load(samples_filename, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Results file not found for rank {rank} at {samples_filename}. Skipping.")
        return None, None, None, None  # Return None for all values

    h = res['h']
    log_g = res['log_g']
    it = res['it'].item()
    update_its = res['update_its'].item()
    log_f = res['log_f'].item()  # Retrieve log_f from the loaded data
    return h, log_g, it, update_its,log_f


num_ranks = 6  # Assuming 6 ranks as per the sbatch file for main simulation

# Create a directory for plots if it doesn't exist
plot_output_dir = './plots'
if not os.path.exists(plot_output_dir):
    os.makedirs(plot_output_dir)
    print(f"Created plot output directory: {plot_output_dir}")

print(f"Searching for results in: ./results")

for rank_val in range(0, num_ranks):
    h, log_g, it, update_its,log_f = load_walker_data(rank_val)  # Use the helper function to load data
    if h is None:  # Check if loading failed
        continue

    print(f"Processing Rank {rank_val} at iteration {it}. Last update: {update_its}")

    # --- FIGURE 1: h and log_g Histograms (Bar Charts) ---
    fig_histograms = plt.figure(figsize=(10, 8))  # Larger figure to accommodate two subplots

    # Subplot for h histogram
    plt.subplot(211)
    plt.bar(np.arange(h.size), h.flatten())
    plt.title(f'Rank {rank_val} - h histogram (Iteration {it}, Last update_f {log_f})')
    plt.xlabel('Combined E/Q Bin Index')  # h is 1D, so it covers all e-q bins for this walker
    plt.ylabel('Count')

    # Subplot for log_g histogram (flattened)
    plt.subplot(212)
    # If log_g is 2D, flattening it will create a 1D array.
    # The x-axis will then be the linear index of the flattened array.
    plt.bar(np.arange(log_g.size), log_g.flatten())
    plt.title(f'Rank {rank_val} - log_g flattened histogram (Iteration {it}, Last update_f {log_f})')
    plt.xlabel('Combined E/Q Bin Index')
    plt.ylabel('Log_g Value')

    plt.tight_layout()  # Adjust layout to prevent overlapping titles/labels
    histograms_plot_filename = os.path.join(plot_output_dir, f'rank_{rank_val}_it_{it}_histograms.png')
    plt.savefig(histograms_plot_filename)
    plt.close(fig_histograms)  # Close the figure to free memory
    print(f"Saved bar chart histograms for rank {rank_val} to {histograms_plot_filename}")

    # --- FIGURE 2: log_g Line Plots (for specific E-Accuracy values) ---
    # Ensure log_g is 2D for this plot (it should be, where first dim is e, second is q)
    if log_g.ndim == 2:
        # 'rang' represents the test accuracy (q) values for this walker's range
        rang = np.arange(eq_limits[rank_val]['q_min'], eq_limits[rank_val]['q_max'] + 1)

        # Determine the actual e-accuracy values from the relative indices
        # Based on config.py, e_min for walkers is usually n_train - 5 (e.g., 295 if n_train=300)
        e_acc_0_idx = eq_limits[rank_val]['e_min'] + 0  # This maps log_g[0,:] to actual e_min
        e_acc_5_idx = eq_limits[rank_val]['e_min'] + 5  # This maps log_g[5,:] to actual e_min + 5

        fig_log_g_lines = plt.figure(figsize=(12, 8))  # A bit larger for two subplots

        # Subplot for log_g[5,:] (which corresponds to e-accuracy 300)
        plt.subplot(211)
        try:
            # log_g is already sub-indexed to the walker's specific e and q range.
            # So, log_g[5, :] directly gives the data for e-index 5 across its q range.
            plt.plot(rang, log_g[5, :])
            plt.title(f'Rank {rank_val} - log_g (E-Acc: {e_acc_5_idx}) (It {it} Last update_f {log_f})')
            plt.xlabel('Test Accuracy (Q)')
            plt.ylabel('Log_g Value')
            plt.grid(True, linestyle='--', alpha=0.7)  # Add grid for readability
        except IndexError:
            print(f"Warning: log_g[5,:] out of bounds for Rank {rank_val}. Skipping this subplot.")
        except Exception as e:
            print(f"An error occurred plotting log_g[5,:] for Rank {rank_val}: {e}")

        # Subplot for log_g[0,:] (which corresponds to e-accuracy 295)
        plt.subplot(212)
        try:
            plt.plot(rang, log_g[0, :])
            plt.title(f'Rank {rank_val} - log_g (E-Acc: {e_acc_0_idx}) (It {it} Last update_f {log_f})')
            plt.xlabel('Test Accuracy (Q)')
            plt.ylabel('Log_g Value')
            plt.grid(True, linestyle='--', alpha=0.7)  # Add grid for readability
        except IndexError:
            print(f"Warning: log_g[0,:] out of bounds for Rank {rank_val}. Skipping this subplot.")
        except Exception as e:
            print(f"An error occurred plotting log_g[0,:] for Rank {rank_val}: {e}")

        plt.tight_layout()  # Adjust layout
        log_g_lines_plot_filename = os.path.join(plot_output_dir, f'rank_{rank_val}_it_{it}_log_g_curves.png')
        plt.savefig(log_g_lines_plot_filename)
        plt.close(fig_log_g_lines)  # Close the figure
        print(f"Saved log_g line plots for rank {rank_val} to {log_g_lines_plot_filename}")
    else:
        print(f"Warning: log_g for rank {rank_val} is not 2D, skipping line plots.")

print("\nPlotting process completed. Check the 'plots' directory for generated images.")


def plot_all_ranks(ei=5):
    """
    Plots the normalized log_g curves for all ranks, stitching them together.
    The function is generalized to work for any number of ranks (qw) and any overlap (L/2).

    Args:
        ei (int): The index of the E (training accuracy) bin to plot.
    """

    # Get the number of ranks from the config
    qw = len(eq_limits)
    if qw < 2:
        print("Not enough ranks to stitch log_g curves.")
        return

    # Get the window size from the config
    L = eq_limits[0]['q_max'] - eq_limits[0]['q_min'] + 1
    overlap = L // 2

    # Load log_g for all ranks
    lgs = {}
    its = {}
    total_found_ranks = 0
    for rank in range(qw):
        samples_filename = f'./results/rank_{rank}_results.npz'
        if not os.path.exists(samples_filename):
            print(f"File not found for rank {rank}: {samples_filename}")
            continue
        res = np.load(samples_filename, allow_pickle=True)
        lgs[rank] = res['log_g'][ei] # Get the log_g for the specified E bin
        its[rank] = res['it'].item() # Get the iteration count
        total_found_ranks += 1

    if total_found_ranks < 2:
        print("Not enough ranks with results files found to perform stitching.")
        return

    # The iteration count will be the same for all ranks at the end
    it = max(its.values())

    # Compute displacement constants and estimation errors
    # We will store them in lists for generalization
    displacements = {}
    errors = []

    for rank in range(qw - 1):
        # Skip if either rank is missing
        if rank not in lgs or (rank + 1) not in lgs:
            continue

        # Overlap region of the current rank's log_g
        current_overlap_lg = lgs[rank][-overlap:]
        # Overlap region of the next rank's log_g
        next_overlap_lg = lgs[rank + 1][:overlap]

        # Compute the displacement constant
        displacement = (current_overlap_lg - next_overlap_lg).mean() # Mean displacement between the two overlaps
        displacements[rank + 1] = displacement # Store displacement for the next rank

        # Compute the estimation error for this overlap
        error = ((current_overlap_lg - next_overlap_lg - displacement) ** 2).mean() # Mean squared error of the overlap
        errors.append(error)

    # Compute cumulative displacements for plotting
    cumulative_displacements = {0: 0.0}
    for rank in range(1, qw):
        cumulative_displacements[rank] = cumulative_displacements[rank - 1] + displacements.get(rank, 0.0)

    # Calculate total error
    total_error = sum(errors) # Sum of all estimation errors

    # Plot the shifted and normalized curves
    all_shifted_lgs = []
    for rank in range(qw):
        if rank in lgs:
            shifted_lg = lgs[rank] + cumulative_displacements[rank] # Apply cumulative displacement
            all_shifted_lgs.append(shifted_lg)

    if not all_shifted_lgs:
        print("No log_g data to plot.")
        return

    # Find the global minimum for normalization
    bottom = np.min(np.concatenate(all_shifted_lgs))

    # --- Start creating the plot ---
    fig = plt.figure(figsize=(12, 8))

    for rank in range(qw):
        if rank in lgs:
            shifted_lg = all_shifted_lgs[rank] - bottom # Normalize by subtracting the global minimum
            q_min = eq_limits[rank]['q_min']
            q_max = eq_limits[rank]['q_max']
            plt.plot(np.arange(q_min, q_max + 1), shifted_lg, label=f'Rank {rank}')

    print(f"Total Estimation Error: {total_error}")
    plt.title(f'Log_g Curves (stitched), E-bin {ei}, Iteration {it}')
    plt.xlabel('Test Accuracy (Q)')
    plt.ylabel('log_g')
    plt.legend()
    plt.grid(True)

    # Save the figure to a file,
    plot_output_dir = './plots'
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)

    plot_filename = os.path.join(plot_output_dir, f'stitched_log_g_e{ei}_it{it}.png')
    plt.savefig(plot_filename)
    plt.close(fig)  # Close the figure to free up memory

    print(f"Saved stitched log_g plot to {plot_filename}")

# Plotting all ranks for a specific E bin (ei)
plot_all_ranks()