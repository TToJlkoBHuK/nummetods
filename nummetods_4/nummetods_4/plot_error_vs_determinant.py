import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('error_vs_determinant.txt')
determinants = data[:, 0]
errors = data[:, 1]

plt.figure(figsize=(8, 6))
plt.loglog(determinants, errors, marker='o')
plt.xlabel('Определитель матрицы')
plt.ylabel('Погрешность решения')
plt.title('Зависимость погрешности от малого определителя')
plt.grid(True, which='both')
plt.show()
