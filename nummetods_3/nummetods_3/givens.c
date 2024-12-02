#include <stdlib.h>
#include <math.h>
#include "givens.h"

void givens_rotation(double** A, double* b, size_t n) {
    for (size_t j = 0; j < n; j++) {
        for (size_t i = n - 1; i > j; i--) {
            double a = A[i - 1][j];
            double b_elem = A[i][j];

            double r = sqrt(a * a + b_elem * b_elem);
            double c = a / r;
            double s = -b_elem / r;

            for (size_t k = j; k < n; k++) {
                double temp = c * A[i - 1][k] - s * A[i][k];
                A[i][k] = s * A[i - 1][k] + c * A[i][k];
                A[i - 1][k] = temp;
            }

            double temp_b = c * b[i - 1] - s * b[i];
            b[i] = s * b[i - 1] + c * b[i];
            b[i - 1] = temp_b;
        }
    }
}

void back_substitution(double** R, double* b, double* x, size_t n) {
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (size_t j = i + 1; j < n; j++) {
            x[i] -= R[i][j] * x[j];
        }
        x[i] /= R[i][i];
    }
}