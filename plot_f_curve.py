import os
import re
import numpy as np
import matplotlib.pyplot as plt

def find_output_file(folder_path, prefix="plot_rewl_", suffix=".out"):
    """
    Find the first REWL output file matching a known pattern.
    """
    for fname in os.listdir(folder_path):
        if fname.startswith(prefix) and fname.endswith(suffix):
            return os.path.join(folder_path, fname)
    raise FileNotFoundError(f"No file starting with {prefix} and ending with {suffix} found in {folder_path}.")


def parse_last_updates(filepath):
    """
    Parse all 'Last update:' entries from a REWL output file.
    Returns a list of dictionaries {f: iteration}.
    """
    updates_per_walker = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("Last update:"):
                dict_str = line[len("Last update:"):].strip()
                try:
                    updates = eval(dict_str)
                    updates_per_walker.append(updates)
                except:
                    print("Skipping malformed line:", line)
    return updates_per_walker


def plot_update_iterations(updates_per_walker):
    """
    Plots log10(iteration) vs. f for each walker.
    """
    plt.figure(figsize=(10, 6))

    for i, updates in enumerate(updates_per_walker):
        f_values = sorted(updates.keys(), reverse=True)
        iterations = [updates[f] for f in f_values]
        log_iterations = [np.log10(it) for it in iterations]
        plt.plot(f_values, log_iterations, label=f'Walker {i}')

    plt.xlabel('Modification factor f')
    plt.ylabel('log10(Iteration of f update)')
    plt.xscale('log')
    plt.title('Log10 of iteration vs. modification factor f')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder = "."  # You can change this to the path where the output file resides
    filepath = find_output_file(folder)
    updates = parse_last_updates(filepath)
    plot_update_iterations(updates)
