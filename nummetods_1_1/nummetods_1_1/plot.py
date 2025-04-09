import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob # To find files easily

# --- Plotting Configuration ---
plt.style.use('default') # Nicer plot style
# Increase default font sizes
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10})
OUTPUT_DIR = "plots" # Directory to save plots

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Plotting Functions ---

def plot_function(data_file, title_prefix):
    """Plots function data from a file."""
    try:
        # Use pandas for robust reading, handling potential 'nan'
        df = pd.read_csv(data_file, sep='\t', na_values=['nan', 'NAN'])
        # Drop rows where y is NaN for plotting continuity, but keep NaNs for gaps
        # df_clean = df.dropna(subset=[df.columns[1]])

        plt.figure(figsize=(10, 6))
        # Plot using original df to show gaps where y was 'nan'
        plt.plot(df[df.columns[0]], df[df.columns[1]], label=f'{title_prefix}')

        # Add a horizontal line at y=0
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

        plt.title(f"Function Plot: {title_prefix}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        # Adjust y-limits if values become very large to focus near zero
        y_min, y_max = plt.ylim()
        # Limit based on non-NaN values maybe
        valid_y = df[df.columns[1]].dropna()
        if not valid_y.empty:
            std_dev = valid_y.std()
            median = valid_y.median()
            # Focus the plot somewhat, avoiding extreme outliers unless necessary
            reasonable_min = max(y_min, median - 5 * std_dev if std_dev > 0 else median - 1)
            reasonable_max = min(y_max, median + 5 * std_dev if std_dev > 0 else median + 1)
             # Ensure we always include y=0
            if reasonable_min > -1: reasonable_min = -1
            if reasonable_max < 1: reasonable_max = 1
            # But don't make limits *too* tight if the range is naturally large
            if y_max > reasonable_max * 2 : reasonable_max = y_max # Keep original max if much larger
            if y_min < reasonable_min * 2 : reasonable_min = y_min # Keep original min if much smaller
            try:
                plt.ylim(reasonable_min, reasonable_max)
            except Exception:
                pass # Ignore ylim errors if values are weird


        plot_filename = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(data_file))[0]}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved function plot: {plot_filename}")
    except FileNotFoundError:
        print(f"Warning: Data file not found: {data_file}")
    except Exception as e:
        print(f"Error plotting {data_file}: {e}")

def plot_error_vs_iteration(files_dict, title_suffix, func_name):
    """Plots absolute error vs iteration for multiple methods."""
    plt.figure(figsize=(10, 6))
    found_data = False
    for method, filename in files_dict.items():
        try:
            df = pd.read_csv(filename, sep='\t')
            if not df.empty and df.columns[1] in df and df[df.columns[1]].notna().any():
                 # Use the first column as iteration, second as error
                plt.plot(df.iloc[:, 0], df.iloc[:, 1], marker='.', linestyle='-', label=method)
                found_data = True
            else:
                 print(f"Warning: No valid data found in {filename}")

        except FileNotFoundError:
            print(f"Warning: Data file not found: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not found_data:
        print(f"Skipping plot for '{title_suffix}' due to missing/empty data.")
        plt.close() # Close the empty figure
        return

    plt.title(f"Convergence: Absolute Error vs. Iteration ({func_name})")
    plt.xlabel("Iteration Number")
    plt.ylabel("Absolute Error |x_k - true_root|")
    plt.yscale('log') # Error typically decreases exponentially
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5) # Grid for log scale
    plot_filename = os.path.join(OUTPUT_DIR, f"Err_vs_Iter_{title_suffix}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved convergence plot: {plot_filename}")


def plot_error_vs_tolerance(files_dict, title_suffix, func_name):
    """Plots achieved error vs requested tolerance."""
    plt.figure(figsize=(10, 6))
    min_tol, max_tol = np.inf, -np.inf
    found_data = False

    for method, filename in files_dict.items():
        try:
            df = pd.read_csv(filename, sep='\t').dropna() # Drop rows where calculation might have failed (NaN)
            if not df.empty:
                 tols = df.iloc[:, 0]
                 errs = df.iloc[:, 1]
                 plt.plot(tols, errs, marker='o', linestyle='-', label=f"{method}")
                 min_tol = min(min_tol, tols.min())
                 max_tol = max(max_tol, tols.max())
                 found_data = True
            else:
                 print(f"Warning: No valid data after dropping NaNs in {filename}")

        except FileNotFoundError:
            print(f"Warning: Data file not found: {filename}")
        except Exception as e:
             print(f"Error processing {filename}: {e}")

    if not found_data:
        print(f"Skipping plot for '{title_suffix}' due to missing/empty data.")
        plt.close()
        return

    # Plot the bisector y=x line
    if min_tol < max_tol:
        line_range = np.array([min_tol * 0.9, max_tol * 1.1])
        plt.plot(line_range, line_range, 'k--', label="Achieved Error = Requested Tolerance (y=x)")


    plt.title(f"Precision: Achieved Error vs. Requested Tolerance ({func_name})")
    plt.xlabel("Requested Tolerance")
    plt.ylabel("Achieved Absolute Error |x_final - true_root|")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    # Ensure axis limits are reasonable for log scale
    plt.xlim(left=min_tol * 0.5 if min_tol > 0 else 1e-17)
    plt.ylim(bottom=plt.ylim()[0] if plt.ylim()[0] > 0 else 1e-17)


    plot_filename = os.path.join(OUTPUT_DIR, f"Err_vs_Tol_{title_suffix}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved precision plot: {plot_filename}")


def plot_perturbation(data_file, title_suffix, func_name):
    """Plots relative error vs perturbation percentage using a box plot."""
    try:
        df = pd.read_csv(data_file, sep='\t').dropna()
        if df.empty:
            print(f"Warning: No valid data after dropping NaNs in {data_file}. Skipping plot.")
            return

        # Use pandas for easier grouping and plotting
        # Ensure the percentage column is treated as categorical for boxplot grouping
        df.iloc[:, 0] = df.iloc[:, 0].astype(str) + '%'

        plt.figure(figsize=(12, 7))
        # Create a boxplot: shows median, quartiles, outliers
        df.boxplot(column=df.columns[1], by=df.columns[0], grid=True)

        plt.title(f"Sensitivity: Relative Error vs. Coefficient Perturbation ({func_name}, Secant)")
        plt.suptitle('') # Remove default suptitle generated by pandas boxplot
        plt.xlabel("Perturbation Percentage Applied to Constant Term")
        plt.ylabel("Relative Error |(perturbed_root - true_root) / true_root|")
        # Optional: Set y-axis to log scale if errors vary widely
        # plt.yscale('log')
        # plt.ylim(bottom=1e-9) # Adjust if using log scale

        plot_filename = os.path.join(OUTPUT_DIR, f"Perturbation_{title_suffix}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved perturbation plot: {plot_filename}")

    except FileNotFoundError:
        print(f"Warning: Data file not found: {data_file}")
    except Exception as e:
        print(f"Error plotting perturbation data from {data_file}: {e}")


# --- Main Execution ---

if __name__ == "__main__":
    print("Starting Python plotting script...")

    # Task 2: Plot Functions
    print("\n--- Plotting Functions ---")
    func_files = glob.glob("f?_data.txt") # Find f1_data.txt, f2_data.txt, etc.
    for f in func_files:
         # Extract function name like 'f1' from 'f1_data.txt'
         func_name = os.path.splitext(os.path.basename(f))[0].split('_')[0]
         plot_function(f, func_name.upper())

    # Extract function names used from file generation (assuming C code used f1, f2, f3)
    # This relies on the C code creating files with names like "err_vs_iter_f1_bisec.txt"
    func_identifiers = set()
    all_data_files = glob.glob("err_vs_*.txt") + glob.glob("perturb_*.txt")
    for f in all_data_files:
        parts = os.path.basename(f).split('_')
        if len(parts) > 2:
            # Assuming format like err_vs_iter_FUNCTION_METHOD.txt or perturb_FUNCTION_METHOD.txt
            func_identifiers.add(parts[-2]) # e.g., 'f1', 'f2', 'f3'

    print(f"\nFound data for functions: {list(func_identifiers)}")

    # Task 3: Plot Error vs. Iteration
    print("\n--- Plotting Error vs. Iteration ---")
    for func_id in func_identifiers:
         files = {
             'Bisection': f"err_vs_iter_{func_id}_bisec.txt",
             'Secant': f"err_vs_iter_{func_id}_secant.txt"
         }
         plot_error_vs_iteration(files, func_id.upper(), func_id.upper())


    # Task 4: Plot Error vs. Tolerance
    print("\n--- Plotting Error vs. Tolerance ---")
    for func_id in func_identifiers:
        files = {
            'Bisection': f"err_vs_tol_{func_id}_bisec.txt",
            'Secant': f"err_vs_tol_{func_id}_secant.txt"
        }
        plot_error_vs_tolerance(files, func_id.upper(), func_id.upper())

    # Task 5: Plot Perturbation Results
    print("\n--- Plotting Perturbation Analysis ---")
    # Assuming perturbation was done only for f2 with secant as per C code structure
    perturb_files = glob.glob("perturb_f?_secant.txt")
    for f in perturb_files:
        func_name = os.path.splitext(os.path.basename(f))[0].split('_')[1] # e.g., 'f2'
        plot_perturbation(f, func_name.upper(), func_name.upper())


    print("\nPython plotting script finished. Plots saved in 'plots' directory.")
