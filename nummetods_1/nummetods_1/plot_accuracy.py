import numpy as np
import matplotlib.pyplot as plt

tolerances = []
errors = []
with open('accuracy_results.txt', 'r') as file:
    for line in file:
        tol, err = line.strip().split()
        tolerances.append(float(tol))
        errors.append(float(err))

plt.figure(figsize=(8,6))
plt.loglog(tolerances, errors, marker='o', label='Полученная погрешность')
plt.plot(tolerances, tolerances, linestyle='--', label='Биссектриса')
plt.xlabel('Заданная точность')
plt.ylabel('Полученная погрешность')
plt.title('Зависимость погрешности от заданной точности')
plt.grid(True, which='both')
plt.legend()
plt.show()
