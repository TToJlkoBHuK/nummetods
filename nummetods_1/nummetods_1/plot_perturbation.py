import matplotlib.pyplot as plt

perturbations = []
relative_errors = []
with open('perturbation_results.txt', 'r') as file:
    for line in file:
        p, err = line.strip().split()
        perturbations.append(float(p))
        relative_errors.append(float(err))

plt.figure(figsize=(8,6))
plt.plot(perturbations, relative_errors, marker='o')
plt.xlabel('Величина возмущения (%)')
plt.ylabel('Относительная погрешность')
plt.title('Зависимость относительной погрешности от возмущения')
plt.grid(True)
plt.show()
