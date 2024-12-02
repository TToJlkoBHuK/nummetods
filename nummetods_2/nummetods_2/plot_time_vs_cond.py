import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('time_vs_cond.txt')
cond_numbers = data[:, 0]
times = data[:, 1]

plt.figure(figsize=(8, 6))
plt.loglog(cond_numbers, times, marker='o')
plt.xlabel('Число обусловленности')
plt.ylabel('Время выполнения (сек)')
plt.title('Зависимость времени выполнения от числа обусловленности')
plt.grid(True, which='both')
plt.show()
