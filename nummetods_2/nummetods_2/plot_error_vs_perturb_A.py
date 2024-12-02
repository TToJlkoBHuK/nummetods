import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('error_vs_perturb_A.txt')
perturbations = data[:, 0]
relative_errors = data[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(perturbations, relative_errors, marker='o')
plt.xlabel('Возмущение элемента матрицы (%)')
plt.ylabel('Относительная погрешность')
plt.title('Зависимость относительной погрешности от возмущения элемента матрицы')
plt.grid(True)
plt.show()
