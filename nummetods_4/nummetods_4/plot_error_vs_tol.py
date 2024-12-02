import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('iterations_vs_tol.txt')
tolerances = data[:, 0]
errors = data[:, 2]

plt.figure(figsize=(8, 6))
plt.loglog(tolerances, errors, marker='o')
plt.xlabel('Заданная точность')
plt.ylabel('Погрешность решения')
plt.title('Зависимость погрешности от заданной точности')
plt.grid(True, which='both')
plt.show()
