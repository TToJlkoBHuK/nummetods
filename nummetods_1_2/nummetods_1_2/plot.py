import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Настройки ---
FILE_COND = "lu_results_cond.txt"
FILE_PERT_B = "lu_results_pert_b.txt"
FILE_PERT_A = "lu_results_pert_A.txt"
OUTPUT_DIR = "lu_plots"

# Создать папку для графиков, если ее нет
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Чтение данных ---
try:
    df_cond = pd.read_csv(FILE_COND, sep=' ', skipinitialspace=True)
    df_pert_b = pd.read_csv(FILE_PERT_B, sep=' ', skipinitialspace=True)
    df_pert_a = pd.read_csv(FILE_PERT_A, sep=' ', skipinitialspace=True)

    # Очистка от возможных NaN (если были ошибки в C)
    df_cond.dropna(inplace=True)
    df_pert_b.dropna(inplace=True)
    df_pert_a.dropna(inplace=True)

    # Замена бесконечных значений на NaN, затем удаление
    df_cond.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_pert_b.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_pert_a.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cond.dropna(inplace=True)
    df_pert_b.dropna(inplace=True)
    df_pert_a.dropna(inplace=True)


except FileNotFoundError:
    print(f"Error: One or more result files not found ({FILE_COND}, {FILE_PERT_B}, {FILE_PERT_A})")
    print("Please run the C code first.")
    exit()
except pd.errors.EmptyDataError:
     print(f"Error: One or more result files are empty.")
     exit()


if df_cond.empty:
    print(f"Error: {FILE_COND} is empty or contains only invalid data.")
    exit()


# --- График 1: Зависимость точности от числа обусловленности ---
plt.figure(figsize=(10, 6))
plt.plot(df_cond['TargetCond'], df_cond['ErrorInf'], marker='o', linestyle='-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Целевое число обусловленности κ(A)')
plt.ylabel('Погрешность решения (||x_exact - x_comp||_inf)')
plt.title('Зависимость точности LU-решения от числа обусловленности')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_vs_cond.png'))
# plt.show()
plt.close()

# --- График 2: Зависимость времени выполнения от числа обусловленности ---
plt.figure(figsize=(10, 6))
plt.plot(df_cond['TargetCond'], df_cond['Time'], marker='s', linestyle='-')
plt.xscale('log')
# plt.yscale('log') # Время может не сильно меняться, логарифм может не подойти
plt.xlabel('Целевое число обусловленности κ(A)')
plt.ylabel('Время выполнения (секунды)')
plt.title('Зависимость времени выполнения LU-разложения от числа обусловленности')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'time_vs_cond.png'))
# plt.show()
plt.close()

# --- График 3: Зависимость относительной погрешности от возмущения вектора b ---
plt.figure(figsize=(10, 6))
# Группируем по проценту возмущения и строим для каждого числа обусловленности
markers = ['o', 's', '^', 'v', 'd', '>', '<', 'p', '*', '+']
cond_nums_b = df_pert_b['TargetCond'].unique()

for i, cond in enumerate(cond_nums_b):
    df_subset = df_pert_b[df_pert_b['TargetCond'] == cond]
    if not df_subset.empty:
        plt.plot(df_subset['Perturbation'], df_subset['RelErrorAvg'],
                 marker=markers[i % len(markers)], linestyle='-',
                 label=f'κ ≈ {cond:.0e}')

plt.xlabel('Возмущение вектора b (%)')
plt.ylabel('Средняя относительная погрешность')
plt.title('Зависимость относительной погрешности от возмущения вектора b')
plt.yscale('log')
plt.xticks(df_pert_b['Perturbation'].unique()) # Устанавливаем метки 1, 2, 3, 4, 5
plt.legend(title="Число обусл.", loc='best', fontsize='small')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rel_error_vs_b_perturbation.png'))
# plt.show()
plt.close()

# --- График 4: Зависимость относительной погрешности от возмущения матрицы A ---
plt.figure(figsize=(12, 7))

# Данные для возмущения минимального элемента (ElementType == 0)
df_pert_a_min = df_pert_a[df_pert_a['ElementType'] == 0]
# Данные для возмущения максимального элемента (ElementType == 1)
df_pert_a_max = df_pert_a[df_pert_a['ElementType'] == 1]

cond_nums_a = df_pert_a['TargetCond'].unique()

# Подграфик для минимального элемента
plt.subplot(1, 2, 1)
for i, cond in enumerate(cond_nums_a):
    df_subset = df_pert_a_min[df_pert_a_min['TargetCond'] == cond]
    if not df_subset.empty:
        plt.plot(df_subset['Perturbation'], df_subset['RelErrorAvg'],
                 marker=markers[i % len(markers)], linestyle='-',
                 label=f'κ ≈ {cond:.0e}')
plt.yscale('log')
plt.xlabel('Возмущение (%)')
plt.ylabel('Средняя отн. погрешность')
plt.title('Возмущение мин. элемента A')
plt.xticks(df_pert_a['Perturbation'].unique())
plt.grid(True, which="both", ls="--")
plt.legend(title="Число обусл.", loc='best', fontsize='small')

# Подграфик для максимального элемента
plt.subplot(1, 2, 2)
for i, cond in enumerate(cond_nums_a):
    df_subset = df_pert_a_max[df_pert_a_max['TargetCond'] == cond]
    if not df_subset.empty:
        plt.plot(df_subset['Perturbation'], df_subset['RelErrorAvg'],
                 marker=markers[i % len(markers)], linestyle='-',
                 label=f'κ ≈ {cond:.0e}')
plt.yscale('log')
plt.xlabel('Возмущение (%)')
# plt.ylabel('Средняя отн. погрешность') # Можно убрать, т.к. ось общая
plt.title('Возмущение макс. элемента A')
plt.xticks(df_pert_a['Perturbation'].unique())
plt.grid(True, which="both", ls="--")
plt.legend(title="Число обусл.", loc='best', fontsize='small')


plt.suptitle('Зависимость относительной погрешности от возмущения элементов матрицы A')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Добавляем отступ для общего заголовка
plt.savefig(os.path.join(OUTPUT_DIR, 'rel_error_vs_A_perturbation.png'))
# plt.show()
plt.close()

print(f"Plots saved to '{OUTPUT_DIR}' directory.")
