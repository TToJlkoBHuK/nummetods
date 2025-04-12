import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_data(filename):
    """Loads data from a two-column txt file, skipping header."""
    try:
        # Use pandas for robust loading, handling potential NaN values
        data = pd.read_csv(filename, delim_whitespace=True, skiprows=1, header=None, names=['x', 'y'], na_values=['NAN', 'nan'])
        # Drop rows where either column is NaN, as they represent non-convergence
        data.dropna(inplace=True)
        if data.empty:
            print(f"Warning: No valid data points found in {filename} after dropping NaNs.")
            return None, None
        return data['x'].values, data['y'].values
    except FileNotFoundError:
        print(f"Error: File not found - {filename}")
        return None, None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None

# --- Plotting Functions ---

def plot_accuracy_vs_cond(filename="results_accuracy_vs_cond.txt"):
    cond, acc = load_data(filename)
    if cond is None or acc is None or len(cond) == 0:
        print("Skipping plot: Accuracy vs Condition Number (no data)")
        return
    plt.figure(figsize=(10, 6))
    plt.loglog(cond, acc, 'o-', label='Final Accuracy vs Condition Number')
    plt.xlabel('Condition Number κ(A)')
    plt.ylabel('Final Accuracy ||x_sol - x_exact||∞')
    plt.title('Simple Iteration: Final Accuracy vs Condition Number (Tol=1e-10)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("plot_accuracy_vs_cond.png")
    print(f"Saved plot: plot_accuracy_vs_cond.png")
    # plt.show()

def plot_time_vs_cond(filename="results_time_vs_cond.txt"):
    cond, time = load_data(filename)
    if cond is None or time is None or len(cond) == 0:
        print("Skipping plot: Time vs Condition Number (no data)")
        return
    plt.figure(figsize=(10, 6))
    plt.loglog(cond, time, 's-', label='Execution Time vs Condition Number')
    plt.xlabel('Condition Number κ(A)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Simple Iteration: Execution Time vs Condition Number (Tol=1e-10)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("plot_time_vs_cond.png")
    print(f"Saved plot: plot_time_vs_cond.png")
    # plt.show()

def plot_error_vs_eps(file_good="results_error_vs_eps_good.txt",
                       file_bad="results_error_vs_eps_bad.txt"):
    eps_good, err_good = load_data(file_good)
    eps_bad, err_bad = load_data(file_bad)

    plt.figure(figsize=(10, 6))
    plot_exists = False
    if eps_good is not None and err_good is not None and len(eps_good) > 0:
        plt.loglog(eps_good, err_good, 'o-', label=f'Good Condition (κ ≈ {10.0:.0e})')
        plot_exists = True
    else:
        print("Skipping Good Condition data in Error vs Epsilon plot (no data)")

    if eps_bad is not None and err_bad is not None and len(eps_bad) > 0:
        plt.loglog(eps_bad, err_bad, 's-', label=f'Bad Condition (κ ≈ {1e5:.0e})')
        plot_exists = True
    else:
        print("Skipping Bad Condition data in Error vs Epsilon plot (no data)")

    if not plot_exists:
        print("Skipping plot: Error vs Epsilon (no valid data)")
        return

    # Plot y=x line for reference (ideal case: final error matches target epsilon)
    min_eps = min(eps_good[0] if eps_good is not None and len(eps_good) > 0 else 1e-1,
                  eps_bad[0] if eps_bad is not None and len(eps_bad) > 0 else 1e-1)
    max_eps = max(eps_good[-1] if eps_good is not None and len(eps_good) > 0 else 1e-15,
                  eps_bad[-1] if eps_bad is not None and len(eps_bad) > 0 else 1e-15)

    ref_eps = np.logspace(np.log10(min_eps), np.log10(max_eps), num=10) # Create reference points
    plt.loglog(ref_eps, ref_eps, 'k--', label='Ideal (Error = Epsilon)')


    plt.xlabel('Target Tolerance (Epsilon)')
    plt.ylabel('Final Achieved Accuracy ||x_sol - x_exact||∞')
    plt.title('Simple Iteration: Final Accuracy vs Target Tolerance')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.gca().invert_xaxis() # Typically plot epsilon decreasing
    plt.savefig("plot_error_vs_eps.png")
    print(f"Saved plot: plot_error_vs_eps.png")
    # plt.show()

def plot_iters_vs_eps(file_good="results_iters_vs_eps_good.txt",
                       file_bad="results_iters_vs_eps_bad.txt"):
    eps_good, iters_good = load_data(file_good)
    eps_bad, iters_bad = load_data(file_bad)

    plt.figure(figsize=(10, 6))
    plot_exists = False
    if eps_good is not None and iters_good is not None and len(eps_good) > 0:
        plt.semilogy(eps_good, iters_good, 'o-', label=f'Good Condition (κ ≈ {10.0:.0e})')
        plot_exists = True
    else:
        print("Skipping Good Condition data in Iterations vs Epsilon plot (no data)")

    if eps_bad is not None and iters_bad is not None and len(eps_bad) > 0:
        plt.semilogy(eps_bad, iters_bad, 's-', label=f'Bad Condition (κ ≈ {1e5:.0e})')
        plot_exists = True
    else:
        print("Skipping Bad Condition data in Iterations vs Epsilon plot (no data)")

    if not plot_exists:
        print("Skipping plot: Iterations vs Epsilon (no valid data)")
        return

    plt.xlabel('Target Tolerance (Epsilon)')
    plt.ylabel('Number of Iterations')
    plt.title('Simple Iteration: Iterations vs Target Tolerance')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.xscale('log') # Log scale for epsilon
    plt.gca().invert_xaxis() # Typically plot epsilon decreasing
    plt.savefig("plot_iters_vs_eps.png")
    print(f"Saved plot: plot_iters_vs_eps.png")
    # plt.show()

def plot_error_vs_iter(file_good="results_error_vs_iter_good.txt",
                       file_bad="results_error_vs_iter_bad.txt"):
    iter_good, err_good = load_data(file_good)
    iter_bad, err_bad = load_data(file_bad)

    plt.figure(figsize=(10, 6))
    plot_exists = False
    if iter_good is not None and err_good is not None and len(iter_good) > 0:
        plt.semilogy(iter_good, err_good, '-', label=f'Good Condition (κ ≈ {10.0:.0e})')
        plot_exists = True
    else:
         print("Skipping Good Condition data in Error vs Iterations plot (no data)")

    if iter_bad is not None and err_bad is not None and len(iter_bad) > 0:
        plt.semilogy(iter_bad, err_bad, '-', label=f'Bad Condition (κ ≈ {1e5:.0e})')
        plot_exists = True
    else:
         print("Skipping Bad Condition data in Error vs Iterations plot (no data)")

    if not plot_exists:
        print("Skipping plot: Error vs Iterations (no valid data)")
        return

    plt.xlabel('Iteration Number (k)')
    plt.ylabel('Error ||x_k - x_exact||∞')
    plt.title('Simple Iteration: Error Convergence over Iterations')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("plot_error_vs_iter.png")
    print(f"Saved plot: plot_error_vs_iter.png")
    # plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    print("Generating plots from C solver results...")

    # Check if result files exist before plotting
    required_files = [
        "results_accuracy_vs_cond.txt", "results_time_vs_cond.txt",
        "results_error_vs_eps_good.txt", "results_error_vs_eps_bad.txt",
        "results_iters_vs_eps_good.txt", "results_iters_vs_eps_bad.txt",
        "results_error_vs_iter_good.txt", "results_error_vs_iter_bad.txt"
    ]
    files_missing = False
    for f in required_files:
        if not os.path.exists(f):
            print(f"Warning: Result file '{f}' not found. Run the C code first.")
            files_missing = True

    if not files_missing:
         print("All result files found.")

    # Generate plots
    plot_accuracy_vs_cond()
    plot_time_vs_cond()
    plot_error_vs_eps()
    plot_iters_vs_eps()
    plot_error_vs_iter()

    print("\nPlotting finished. Check for .png files.")
    plt.show() # Display plots interactively after saving
