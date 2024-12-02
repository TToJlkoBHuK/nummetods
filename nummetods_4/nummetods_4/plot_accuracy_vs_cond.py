import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('accuracy_vs_cond.txt')
cond_numbers = data[:, 0]
errors = data[:, 1]

plt.figure(figsize=(8, 6))
plt.loglog(cond_numbers, errors, marker='o')
plt.xlabel('Число обусловленности')
plt.ylabel('Погрешность решения')
plt.title('Зависимость точности от числа обусловленности')
plt.grid(True, which='both')
plt.show()
