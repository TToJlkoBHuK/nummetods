import numpy as np
import matplotlib.pyplot as plt

# Для хорошей обусловленности
#data_good = np.loadtxt('error_progress_good.txt')
#iterations_good = data_good[:, 0]
#errors_good = data_good[:, 1]

# Для плохой обусловленности
data_bad = np.loadtxt('error_progress_bad.txt')
iterations_bad = data_bad[:, 0]
errors_bad = data_bad[:, 1]

##plt.figure(figsize=(8, 6))
###plt.semilogy(iterations_good, errors_good, label='Хорошая обусловленность')
##plt.semilogy(iterations_bad, errors_bad, label='Плохая обусловленность')
##plt.xlabel('Номер итерации')
##plt.ylabel('Погрешность')
##plt.title('Уменьшение погрешности с ходом итераций')
##plt.legend()
##plt.grid(True, which='both')
##plt.show()

# Используем логарифмы для избежания переполнения
log_errors_bad = np.log(errors_bad)

plt.figure(figsize=(8, 6))
plt.plot(iterations_bad, log_errors_bad, label='Плохая обусловленность (лог шкала)')
plt.xlabel('Номер итерации')
plt.ylabel('Логарифм погрешности')
plt.title('Уменьшение погрешности с ходом итераций (лог шкала)')
plt.legend()
plt.grid(True, which='both')
plt.show()
