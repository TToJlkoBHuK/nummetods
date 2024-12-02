#include <stdlib.h>
#include <math.h>
#include "jacobi.h"
#include "utils.h"

int jacobi_method(double** A, double* b, double* x, size_t n, double tol, int max_iter, double* errors) {
    double* x_old = (double*)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        for (size_t i = 0; i < n; i++) {
            x_old[i] = x[i];
        }

        for (size_t i = 0; i < n; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < n; j++) {
                if (i != j) {
                    sum += A[i][j] * x_old[j];
                }
            }
            x[i] = (b[i] - sum) / A[i][i];
        }

        double error = vector_norm_diff(x, x_old, n);
        errors[iter] = error;

        if (error < tol) {
            free(x_old);
            return iter + 1;
        }
    }
    free(x_old);
    return max_iter;
}