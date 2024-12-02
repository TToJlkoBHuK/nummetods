import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('iterations_vs_tol.txt')
tolerances = data[:, 0]
iterations = data[:, 1]

plt.figure(figsize=(8, 6))
plt.loglog(tolerances, iterations, marker='o')
plt.xlabel('Заданная точность')
plt.ylabel('Число итераций')
plt.title('Зависимость числа итераций от заданной точности')
plt.grid(True, which='both')
plt.show()
