import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
RESULTS_FILE = "results.txt"
HISTORY_GOOD_COND_FILE = "history_cond_1e+01.txt" # Adjust if C code output name differs
HISTORY_BAD_COND_FILE = "history_cond_1e+06.txt"  # Adjust if C code output name differs
OUTPUT_DIR = "plots"

# Define "good" and "bad" condition numbers based on the tested range
GOOD_COND_THRESHOLD = 1e3
BAD_COND_THRESHOLD = 1e5

# --- Create output directory ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Load Data ---
try:
    df = pd.read_csv(RESULTS_FILE, delim_whitespace=True)
    print(f"Successfully loaded {RESULTS_FILE}")
    # print("Data Head:\n", df.head())
    # print("\nData Types:\n", df.dtypes)
    # print("\nCondition Numbers Found:\n", df['ConditionNumber'].unique())

except FileNotFoundError:
    print(f"Error: {RESULTS_FILE} not found. Run the C code first.")
    exit()
except Exception as e:
    print(f"Error reading {RESULTS_FILE}: {e}")
    exit()


# 1) Dependence of accuracy (ActualError) on condition number
plt.figure(figsize=(10, 6))
# Plot average error for each condition number for clarity, or plot all points
# Let's plot points colored by tolerance
unique_tols = sorted(df['TargetTolerance'].unique(), reverse=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_tols)))

for i, tol in enumerate(unique_tols):
    subset = df[df['TargetTolerance'] == tol]
    subset_converged = subset[subset['ActualError'] < 0.5]
    if not subset_converged.empty:
        plt.plot(subset_converged['ConditionNumber'], subset_converged['ActualError'],
                 label=f'Tol={tol:.1e}', alpha=0.7, marker='o', linestyle='-', color=colors[i])

plt.xlabel("Condition Number (Approx.)")
plt.ylabel("Actual Error (||x - x_exact||_inf)")
plt.xscale('log')
plt.yscale('log')
plt.title("Actual Error vs. Condition Number for Different Tolerances")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
plt.savefig(os.path.join(OUTPUT_DIR, "1_error_vs_cond_num.png"))
plt.close()

# 2) Dependence of execution time on condition number
plt.figure(figsize=(10, 6))
# Average time per condition number might be more informative
avg_time = df.groupby('ConditionNumber')['Time'].mean()
plt.plot(avg_time.index, avg_time.values, marker='o', linestyle='-')
# Add scatter for individual points to see variance
# plt.scatter(df['ConditionNumber'], df['Time'], alpha=0.5, s=10)

plt.xlabel("Condition Number (Approx.)")
plt.ylabel("Average Execution Time (s)")
plt.xscale('log')
plt.yscale('log') # Time often increases significantly
plt.title("Average Execution Time vs. Condition Number")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_time_vs_cond_num.png"))
plt.close()

# Separate data for good and bad condition numbers
df_good = df[df['ConditionNumber'] <= GOOD_COND_THRESHOLD]
df_bad = df[df['ConditionNumber'] >= BAD_COND_THRESHOLD]

# Select one representative "good" and "bad" condition number for clarity if needed
good_cond_rep = df_good['ConditionNumber'].median() if not df_good.empty else df['ConditionNumber'].min()
bad_cond_rep = df_bad['ConditionNumber'].median() if not df_bad.empty else df['ConditionNumber'].max()

df_good_rep = df[df['ConditionNumber'] == df[df['ConditionNumber'] <= GOOD_COND_THRESHOLD]['ConditionNumber'].max()] # Representative good
df_bad_rep = df[df['ConditionNumber'] == df[df['ConditionNumber'] >= BAD_COND_THRESHOLD]['ConditionNumber'].min()] # Representative bad

if df_good_rep.empty: df_good_rep = df[df['ConditionNumber'] == df['ConditionNumber'].min()]
if df_bad_rep.empty: df_bad_rep = df[df['ConditionNumber'] == df['ConditionNumber'].max()]


# 3) Dependence of error (ActualError) on target tolerance for good/bad cond.
plt.figure(figsize=(10, 6))
if not df_good_rep.empty:
    # Сортировка данных по возрастанию TargetTolerance
    df_good_rep_sorted = df_good_rep.sort_values('TargetTolerance')
    plt.plot(df_good_rep_sorted['TargetTolerance'], df_good_rep_sorted['ActualError'], 
             marker='o', linestyle='-', label=f'Good Cond (~{df_good_rep["ConditionNumber"].iloc[0]:.1e})')
    
if not df_bad_rep.empty:
    df_bad_converged = df_bad_rep[df_bad_rep['ActualError'] < 0.5]
    if not df_bad_converged.empty:
        # Сортировка данных по возрастанию TargetTolerance
        df_bad_converged_sorted = df_bad_converged.sort_values('TargetTolerance')
        plt.plot(df_bad_converged_sorted['TargetTolerance'], df_bad_converged_sorted['ActualError'], 
                 marker='x', linestyle='--', label=f'Bad Cond (~{df_bad_rep["ConditionNumber"].iloc[0]:.1e})')

plt.plot([min(df['TargetTolerance']), max(df['TargetTolerance'])], 
         [min(df['TargetTolerance']), max(df['TargetTolerance'])], 'k:', label='Ideal (Error = Tol)')

plt.xlabel("Target Tolerance")
plt.ylabel("Actual Error (||x - x_exact||_inf)")
plt.xscale('log')
plt.yscale('log')
plt.title("Actual Error vs. Target Tolerance")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
# Убрано invert_xaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_error_vs_tolerance.png"))
plt.close()

# 4) Dependence of iterations on target tolerance for good/bad cond.
plt.figure(figsize=(10, 6))
if not df_good_rep.empty:
    # Сортировка данных по возрастанию TargetTolerance
    df_good_rep_sorted = df_good_rep.sort_values('TargetTolerance')
    plt.plot(df_good_rep_sorted['TargetTolerance'], df_good_rep_sorted['Iterations'], 
             marker='o', linestyle='-', label=f'Good Cond (~{df_good_rep["ConditionNumber"].iloc[0]:.1e})')

if not df_bad_rep.empty:
    # Сортировка данных по возрастанию TargetTolerance
    df_bad_rep_sorted = df_bad_rep.sort_values('TargetTolerance')
    plt.plot(df_bad_rep_sorted['TargetTolerance'], df_bad_rep_sorted['Iterations'], 
             marker='x', linestyle='--', label=f'Bad Cond (~{df_bad_rep["ConditionNumber"].iloc[0]:.1e})')

plt.xlabel("Target Tolerance")
plt.ylabel("Number of Iterations")
plt.xscale('log')
plt.yscale('log')
plt.title("Iterations vs. Target Tolerance")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.axhline(df['Iterations'].max(), color='r', linestyle=':', label=f'Max Iterations ({df["Iterations"].max()})')
plt.legend()
# Убрано invert_xaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_iterations_vs_tolerance.png"))
plt.close()

# 5) Decrease in error with iterations for good/bad cond.
plt.figure(figsize=(10, 6))
loaded_history = False
try:
    hist_good = pd.read_csv(HISTORY_GOOD_COND_FILE, delim_whitespace=True, names=['Iteration', 'Error'])
    plt.plot(hist_good['Iteration'], hist_good['Error'], marker='.', linestyle='-', label=f'Good Cond (~{float(HISTORY_GOOD_COND_FILE.split("_")[-1].replace(".txt","")):.1e})')
    loaded_history = True
except FileNotFoundError:
    print(f"Warning: {HISTORY_GOOD_COND_FILE} not found. Skipping part of plot 5.")
except Exception as e:
    print(f"Error reading {HISTORY_GOOD_COND_FILE}: {e}")

try:
    hist_bad = pd.read_csv(HISTORY_BAD_COND_FILE, delim_whitespace=True, names=['Iteration', 'Error'])
     # Limit iterations shown for bad case if it runs for very long without much progress
    max_iter_plot = min(hist_bad['Iteration'].max(), df['Iterations'].max() * 1.1) # Show slightly beyond max iter if reached
    hist_bad_plot = hist_bad[hist_bad['Iteration'] <= max_iter_plot]
    plt.plot(hist_bad_plot['Iteration'], hist_bad_plot['Error'], marker='.', linestyle='-', label=f'Bad Cond (~{float(HISTORY_BAD_COND_FILE.split("_")[-1].replace(".txt","")):.1e})')
    loaded_history = True
except FileNotFoundError:
    print(f"Warning: {HISTORY_BAD_COND_FILE} not found. Skipping part of plot 5.")
except Exception as e:
    print(f"Error reading {HISTORY_BAD_COND_FILE}: {e}")

if loaded_history:
    plt.xlabel("Iteration Number")
    plt.ylabel("Actual Error (||x_k - x_exact||_inf)")
    # plt.xscale('log') # Optional: If convergence is very fast initially
    plt.yscale('log')
    plt.title("Error Reduction per Iteration")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_error_vs_iteration.png"))
else:
    print("Skipping plot 5 due to missing history files.")

plt.close()
