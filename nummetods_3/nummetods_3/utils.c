#include <stdlib.h>
#include <math.h>
#include "utils.h"

double** allocate_matrix(size_t n) {
    double** A = (double**)malloc(n * sizeof(double*));
    for (size_t i = 0; i < n; i++) {
        A[i] = (double*)malloc(n * sizeof(double));
    }
    return A;
}

void free_matrix(double** A, size_t n) {
    for (size_t i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
}

void random_orthogonal_matrix(double** A, size_t n) {
    double** Q = allocate_matrix(n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            Q[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }

    for (size_t k = 0; k < n; k++) {
        for (size_t i = 0; i < k; i++) {
            double dot = 0.0;
            for (size_t j = 0; j < n; j++) {
                dot += Q[j][k] * Q[j][i];
            }
            for (size_t j = 0; j < n; j++) {
                Q[j][k] -= dot * Q[j][i];
            }
        }
        double norm = 0.0;
        for (size_t j = 0; j < n; j++) {
            norm += Q[j][k] * Q[j][k];
        }
        norm = sqrt(norm);
        for (size_t j = 0; j < n; j++) {
            Q[j][k] /= norm;
        }
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A[i][j] = Q[i][j];
        }
    }

    free_matrix(Q, n);
}

void transpose_and_multiply(double** A, double** AtA, size_t n) {
    // AtA = A^T * A
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            AtA[i][j] = 0.0;
            for (size_t k = 0; k < n; k++) {
                AtA[i][j] += A[k][i] * A[k][j];
            }
        }
    }
}

void invert_matrix(double** A, double** A_inv, size_t n) {
    double** aug = allocate_matrix(n);
    for (size_t i = 0; i < n; i++) {
        aug[i] = (double*)malloc(2 * n * sizeof(double));
        for (size_t j = 0; j < n; j++) {
            aug[i][j] = A[i][j];
            aug[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (size_t i = 0; i < n; i++) {
        double max = fabs(aug[i][i]);
        size_t max_row = i;
        for (size_t k = i + 1; k < n; k++) {
            if (fabs(aug[k][i]) > max) {
                max = fabs(aug[k][i]);
                max_row = k;
            }
        }
        if (max_row != i) {
            double* temp = aug[i];
            aug[i] = aug[max_row];
            aug[max_row] = temp;
        }
        double diag = aug[i][i];
        for (size_t j = 0; j < 2 * n; j++) {
            aug[i][j] /= diag;
        }
        for (size_t k = 0; k < n; k++) {
            if (k != i) {
                double factor = aug[k][i];
                for (size_t j = 0; j < 2 * n; j++) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A_inv[i][j] = aug[i][j + n];
        }
        free(aug[i]);
    }
    free(aug);
}

double matrix_infinity_norm(double** A, size_t n) {
    double max = 0.0;
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            sum += fabs(A[i][j]);
        }
        if (sum > max) {
            max = sum;
        }
    }
    return max;
}

double vector_infinity_norm(double* v1, double* v2, size_t n) {
    double max = fabs(v1[0] - v2[0]);
    for (size_t i = 1; i < n; i++) {
        double diff = fabs(v1[i] - v2[i]);
        if (diff > max) {
            max = diff;
        }
    }
    return max;
}