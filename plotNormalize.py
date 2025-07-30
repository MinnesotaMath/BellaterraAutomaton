import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_eigenvalue_histogram(csv_file, output_file, bins, subintervals):
    df = pd.read_csv(csv_file, header=0)
    eigenvalues = df.values.astype(float).flatten() / 3
    eigenvalues = np.sort(eigenvalues)
    
    max_eigen = eigenvalues[-2]
    min_eigen = eigenvalues[1]
    limit = (2 * np.sqrt(2) / 3)

    plt.figure(figsize=(10, 6))
    counts, bin_edges, _ = plt.hist(eigenvalues, bins=bins, edgecolor='black', alpha=0.7, label="Eigenvalues")
    
    most_common_bin_idx = np.argmax(counts)
    most_common_value = (bin_edges[most_common_bin_idx] + bin_edges[most_common_bin_idx + 1]) / 2
    most_common_count = counts[most_common_bin_idx]
    
    plt.axvline(max_eigen, color='red', linestyle='dashed', linewidth=2, label=f"Max: {max_eigen:.2f}")
    plt.axvline(min_eigen, color='blue', linestyle='dashed', linewidth=2, label=f"Min: {min_eigen:.2f}")
    plt.axvline(limit, color='green', linestyle='dotted', linewidth=2, label=r'$\frac{2\sqrt{2}}{3}$')
    plt.axvline(-limit, color='purple', linestyle='dotted', linewidth=2, label=r'$-\frac{2\sqrt{2}}{3}$')
    
    plt.axhline(most_common_count, color='orange', linestyle='solid', linewidth=2, label=f"Most Common Bin: {most_common_value:.2f}, Count: {most_common_count:.0f}")
    
    plt.xlabel('Normalized Eigenvalue')
    plt.ylabel('Occurrences')
    plt.title(f'Histogram of Normalized Eigenvalues in {subintervals} Subintervals')
    plt.xlim(-1, 1)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    # Automatically detect script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir
    csv_dir = os.path.join(base_dir, 'eigenCSV')
    base_output_dir = os.path.join(base_dir, 'eigenPlots')

    os.makedirs(base_output_dir, exist_ok=True)

    for subintervals in range(100, 1100, 100):
        # Edit the directory paths as needed
        output_dir = os.path.join(base_output_dir, f'intervals{subintervals}')
        os.makedirs(output_dir, exist_ok=True)

        for n in range(1, 12):
            csv_file = os.path.join(csv_dir, f"eigenvalues_{n}.csv")
            output_file = os.path.join(output_dir, f"eigenvalues_n={n}_i={subintervals}.png")
            bins = np.linspace(-1, 1, subintervals+1)
            
            if os.path.exists(csv_file):
                plot_eigenvalue_histogram(csv_file, output_file, bins, subintervals)
            else:
                print(f"Warning: {csv_file} not found, skipping.")
