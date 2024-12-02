import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return 0.1 * x**2 - x * np.log(x)

def f2(x):
    return x**4 - 3 * x**2 + 75 * x - 9999

def f3(x):
    y = np.where(np.abs(x - 2.0) < 1e-8, np.nan, (x - 2.0) * (x + 3.0))
    return y

# График для f1
x = np.linspace(1, 2, 500)
plt.figure(figsize=(8, 6))
plt.plot(x, f1(x), label='f1(x)')
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.title('График функции f1(x)')
plt.legend()
plt.grid(True)
plt.show()

# График для f2
x = np.linspace(5, 10, 1000)
plt.figure(figsize=(8, 6))
plt.plot(x, f2(x), label='f2(x)')
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f2(x)')
plt.title('График функции f2(x)')
plt.legend()
plt.grid(True)
plt.show()

# График для f3
x = np.linspace(0, 5, 1000)
plt.figure(figsize=(8, 6))
plt.plot(x, f3(x), label='f3(x)')
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f3(x)')
plt.title('График функции f3(x) с разрывом')
plt.legend()
plt.grid(True)
plt.show()
