#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "functions.h"
#include "methods.h"

#define MAX_ITER 1000

int main() {
    srand(time(NULL)); // random num

    FuncPtr f = f1; // var f1, f2, f3
    double a = 1.0, b = 2.0; // [a, b] f1
    double tol = 1e-15;
    int iter_count;
    double errors[MAX_ITER];

    // 2) graf in python for find value

    // 3) pogresh
    double root = bisection(f, a, b, tol, MAX_ITER, &iter_count, errors);

    // data
    FILE* file = fopen("errors_bisection.txt", "w");
    for (int i = 0; i < iter_count; i++) {
        fprintf(file, "%d %e\n", i + 1, errors[i]);
    }
    fclose(file);

    // analog hord
    root = secant(f, a, b, tol, MAX_ITER, &iter_count, errors);
    file = fopen("errors_secant.txt", "w");
    for (int i = 0; i < iter_count; i++) {
        fprintf(file, "%d %e\n", i + 1, errors[i]);
    }
    fclose(file);

    // 4) tochnost
    double tolerances[] = { 1e-15, 1e-14, 1e-13, 1e-12, 1e-11,
                           1e-10, 1e-9, 1e-8, 1e-7, 1e-6,
                           1e-5, 1e-4, 1e-3, 1e-2, 1e-1 };
    int num_tol = sizeof(tolerances) / sizeof(tolerances[0]);
    file = fopen("accuracy_results.txt", "w");
    for (int i = 0; i < num_tol; i++) {
        tol = tolerances[i];
        root = bisection(f, a, b, tol, MAX_ITER, &iter_count, errors);
        fprintf(file, "%e %e\n", tol, fabs(f(root)));
    }
    fclose(file);

    // 5) vozmushenie
    double perturbations[] = { 0.01, 0.02, 0.03, 0.04, 0.05 };
    int num_perturbations = sizeof(perturbations) / sizeof(perturbations[0]);
    double original_root = bisection(f, a, b, 1e-15, MAX_ITER, &iter_count, errors);

    file = fopen("perturbation_results.txt", "w");
    for (int i = 0; i < num_perturbations; i++) {
        double perturbation = perturbations[i];
        double root_perturbed = bisection(f, a * (1 + perturbation), b * (1 + perturbation), 1e-15, MAX_ITER, &iter_count, errors);
        double relative_error = fabs(root_perturbed - original_root) / fabs(original_root);
        fprintf(file, "%f %e\n", perturbation * 100, relative_error);
    }
    fclose(file);

    return 0;
}