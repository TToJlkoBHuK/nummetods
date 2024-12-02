#include <stdlib.h>
#include "lu.h"

void lu_decomposition(double** A, double** L, double** U, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t k = i; k < n; k++) {
            double sum = 0.0;
            for (size_t j = 0; j < i; j++) {
                sum += L[i][j] * U[j][k];
            }
            U[i][k] = A[i][k] - sum;
        }

        for (size_t k = i; k < n; k++) {
            if (i == k)
                L[i][i] = 1.0;
            else {
                double sum = 0.0;
                for (size_t j = 0; j < i; j++) {
                    sum += L[k][j] * U[j][i];
                }
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
}

void lu_solve(double** L, double** U, double* b, double* x, size_t n) {
    double* y = (double*)malloc(n * sizeof(double));

    // L * y = b
    for (size_t i = 0; i < n; i++) {
        y[i] = b[i];
        for (size_t j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j];
        }
        y[i] /= L[i][i];
    }

    // U * x = y
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (size_t j = i + 1; j < n; j++) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }

    free(y);
}