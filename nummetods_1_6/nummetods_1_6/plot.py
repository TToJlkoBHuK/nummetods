import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Plotting Functions ---

def plot_separability(filename="separability_error.txt"):
    """Plots errors based on eigenvalue separability."""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    try:
        data = pd.read_csv(filename, sep='\t')
        if data.empty:
            print(f"No data found in {filename}")
            return

        # Замена нулевых или отрицательных значений на минимальное положительное
        data['EigenvalueError'] = data['EigenvalueError'].apply(lambda x: x if x > 0 else 1e-10)
        data['EigenvectorError'] = data['EigenvectorError'].apply(lambda x: x if x > 0 else 1e-10)

        bar_width = 0.35
        x = np.arange(len(data['SeparabilityType']))

        fig, ax1 = plt.subplots(figsize=(8, 6))

        color1 = 'tab:red'
        ax1.set_xlabel('Separability Type')
        ax1.set_ylabel('Eigenvalue Error', color=color1)
        bars1 = ax1.bar(x - bar_width/2, data['EigenvalueError'], bar_width, label='Eigenvalue Error', color=color1, log=True)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(data['SeparabilityType'])
        ax1.set_yscale('log') if data['EigenvalueError'].min() > 0 else None  # Проверка перед установкой логарифма

        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Eigenvector Error', color=color2)
        bars2 = ax2.bar(x + bar_width/2, data['EigenvectorError'], bar_width, label='Eigenvector Error', color=color2, log=True)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_yscale('log') if data['EigenvectorError'].min() > 0 else None

        plt.title('Error vs. Eigenvalue Separability (Target EV ≈ 2.0)')
        # Добавление текста
        for i, count in enumerate(data['Iterations']):
            ax1.text(x[i] - bar_width/2, data['EigenvalueError'][i]/2, f'{count} iters', 
                     ha='center', va='top', color='white', fontsize=9, fontweight='bold')
            ax2.text(x[i] + bar_width/2, data['EigenvectorError'][i]/2, f'{count} iters', 
                     ha='center', va='top', color='white', fontsize=9, fontweight='bold')

        fig.tight_layout(pad=3.0)  # Увеличение отступа
        plt.savefig("plot_1_separability_error.png")
        print(f"Generated plot_1_separability_error.png")

    except Exception as e:
        print(f"Error plotting {filename}: {e}")


def plot_tolerance_error(filename="tolerance_error.txt"):
    """Plots errors vs. target tolerance."""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    try:
        data = pd.read_csv(filename, sep='\t')
        if data.empty:
            print(f"No data found in {filename}")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(data['Tolerance'], data['EigenvalueError'], 'o-', label='Eigenvalue Error')
        plt.plot(data['Tolerance'], data['EigenvectorError'], 's--', label='Eigenvector Error')

        plt.xscale('log')
        plt.yscale('log')
        plt.gca().invert_xaxis() # Tolerance decreases left-to-right typically
        plt.xlabel('Target Tolerance (Residual Norm)')
        plt.ylabel('Achieved Error')
        plt.title('Achieved Error vs. Target Tolerance (Good Separability)')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig("plot_2_tolerance_error.png")
        print(f"Generated plot_2_tolerance_error.png")
        # plt.show()

    except Exception as e:
        print(f"Error plotting {filename}: {e}")


def plot_iterations_tolerance(filename="iterations_tolerance.txt"):
    """Plots iterations vs. target tolerance for good/bad separability."""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    try:
        data = pd.read_csv(filename, sep='\t')
        if data.empty:
            print(f"No data found in {filename}")
            return

        data_good = data[data['SeparabilityType'] == 'Good']
        data_bad = data[data['SeparabilityType'] == 'Bad']

        plt.figure(figsize=(10, 6))
        plt.plot(data_good['Tolerance'], data_good['Iterations'], 'o-', label='Good Separability')
        plt.plot(data_bad['Tolerance'], data_bad['Iterations'], 's--', label='Bad Separability')

        plt.xscale('log')
        # plt.yscale('log') # Iterations might not vary enough for log scale
        plt.gca().invert_xaxis()
        plt.xlabel('Target Tolerance (Residual Norm)')
        plt.ylabel('Number of Iterations')
        plt.title('Iterations vs. Target Tolerance')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig("plot_3_iterations_tolerance.png")
        print(f"Generated plot_3_iterations_tolerance.png")
        # plt.show()

    except Exception as e:
        print(f"Error plotting {filename}: {e}")


def plot_error_history(filename="error_vs_iteration.txt"):
    """Plots residual norm error vs. iteration number."""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    try:
        data = pd.read_csv(filename, sep='\t')
        if data.empty:
            print(f"No data found in {filename}")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(data['Iteration'], data['ResidualNorm'], '.-')

        plt.yscale('log')
        plt.xlabel('Iteration Number')
        plt.ylabel('Residual Norm ||Ax - λx||')
        plt.title('Convergence History (Residual Norm vs. Iteration)')
        plt.grid(True, which="both", ls="--")
        plt.savefig("plot_4_error_vs_iteration.png")
        print(f"Generated plot_4_error_vs_iteration.png")
        # plt.show()

    except Exception as e:
        print(f"Error plotting {filename}: {e}")


def plot_perturbation_error(filename="perturbation_error.txt"):
    """Plots eigenvalue/eigenvector error vs. matrix perturbation level."""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    try:
        data = pd.read_csv(filename, sep='\t')
        if data.empty:
            print(f"No data found in {filename}")
            return

        levels = data['PerturbationLevel'].unique() * 100
        levels.sort()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

        # Исправление параметра labels на tick_labels
        ev_data = [data[data['PerturbationLevel'] == lv/100]['EigenvalueError'] for lv in levels]
        ax1.boxplot(ev_data, tick_labels=[f"{lv:.1f}%" for lv in levels], showfliers=False)
        ax1.set_yscale('log')
        ax1.set_ylabel('Eigenvalue Error')
        ax1.set_title('Eigenvalue Error vs. Perturbation')

        vec_data = [data[data['PerturbationLevel'] == lv/100]['EigenvectorError'] for lv in levels]
        ax2.boxplot(vec_data, tick_labels=[f"{lv:.1f}%" for lv in levels], showfliers=False)
        ax2.set_yscale('log')
        ax2.set_ylabel('Eigenvector Error')
        ax2.set_title('Eigenvector Error vs. Perturbation')

        fig.supxlabel('Matrix Perturbation Level (%)')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("plot_5_perturbation_error.png")
        print(f"Generated plot_5_perturbation_error.png")

    except Exception as e:
        print(f"Error plotting {filename}: {e}")

def plot_shift_comparison(filename="shift_comparison.txt"):
    """Compares constant vs variable shifts for good/bad separability."""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    try:
        data = pd.read_csv(filename, sep='\t')
        if data.empty:
            print(f"No data found in {filename}")
            return

        # Замена нулевых значений
        data['EigenvalueError'] = data['EigenvalueError'].apply(lambda x: x if x > 0 else 1e-10)
        data['EigenvectorError'] = data['EigenvectorError'].apply(lambda x: x if x > 0 else 1e-10)

        pivot_iters = data.pivot(index='SeparabilityType', columns='ShiftType', values='Iterations')
        pivot_ev_err = data.pivot(index='SeparabilityType', columns='ShiftType', values='EigenvalueError')
        pivot_vec_err = data.pivot(index='SeparabilityType', columns='ShiftType', values='EigenvectorError')

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        pivot_iters.plot(kind='bar', ax=axes[0], rot=0)
        axes[0].set_title('Iterations Comparison')

        # Проверка наличия положительных значений перед логарифмированием
        if (pivot_ev_err > 0).all().all():
            pivot_ev_err.plot(kind='bar', ax=axes[1], rot=0, logy=True)
        else:
            pivot_ev_err.plot(kind='bar', ax=axes[1], rot=0)
        axes[1].set_title('Eigenvalue Error Comparison')

        if (pivot_vec_err > 0).all().all():
            pivot_vec_err.plot(kind='bar', ax=axes[2], rot=0, logy=True)
        else:
            pivot_vec_err.plot(kind='bar', ax=axes[2], rot=0)
        axes[2].set_title('Eigenvector Error Comparison')

        fig.suptitle('Comparison of Constant vs. Variable Shifts')
        fig.tight_layout()
        plt.savefig("plot_6_shift_comparison.png")
        print(f"Generated plot_6_shift_comparison.png")

    except Exception as e:
        print(f"Error plotting {filename}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    print("Generating plots from C output files...")

    plot_separability()
    plot_tolerance_error()
    plot_iterations_tolerance()
    plot_error_history()
    plot_perturbation_error()
    plot_shift_comparison()

    print("Plot generation complete.")
