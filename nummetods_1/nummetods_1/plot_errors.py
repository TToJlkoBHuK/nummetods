import matplotlib.pyplot as plt

# Метод половинного деления
iterations = []
errors = []
with open('errors_bisection.txt', 'r') as file:
    for line in file:
        n, err = line.strip().split()
        iterations.append(int(n))
        errors.append(float(err))

plt.figure(figsize=(8,6))
plt.semilogy(iterations, errors, marker='o', label='Метод половинного деления')
plt.xlabel('Номер итерации')
plt.ylabel('Абсолютная ошибка')
plt.title('Погрешность от номера итерации (Метод половинного деления)')
plt.grid(True, which='both')
plt.legend()
plt.show()

# Метод хорд
iterations = []
errors = []
with open('errors_secant.txt', 'r') as file:
    for line in file:
        n, err = line.strip().split()
        iterations.append(int(n))
        errors.append(float(err))

plt.figure(figsize=(8,6))
plt.semilogy(iterations, errors, marker='o', color='red', label='Метод хорд')
plt.xlabel('Номер итерации')
plt.ylabel('Абсолютная ошибка')
plt.title('Погрешность от номера итерации (Метод хорд)')
plt.grid(True, which='both')
plt.legend()
plt.show()
