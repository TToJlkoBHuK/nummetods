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

void copy_matrix(double** src, double** dest, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

void transpose_matrix(double** A, double** At, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            At[j][i] = A[i][j];
        }
    }
}

void multiply_matrices(double** A, double** B, double** C, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (size_t k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

double matrix_norm(double** A, size_t n) {
    double norm = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            norm += A[i][j] * A[i][j];
        }
    }
    return sqrt(norm);
}

double vector_norm(double* v, size_t n) {
    double max = fabs(v[0]);
    for (size_t i = 1; i < n; i++) {
        if (fabs(v[i]) > max) {
            max = fabs(v[i]);
        }
    }
    return max;
}

double vector_norm_diff(double* v1, double* v2, size_t n) {
    double max = fabs(v1[0] - v2[0]);
    for (size_t i = 1; i < n; i++) {
        double diff = fabs(v1[i] - v2[i]);
        if (diff > max) {
            max = diff;
        }
    }
    return max;
}

void invert_matrix(double** A, double** A_inv, size_t n) {
    double** aug = (double**)malloc(n * sizeof(double*));
    for (size_t i = 0; i < n; i++) {
        aug[i] = (double*)malloc(2 * n * sizeof(double));
        for (size_t j = 0; j < n; j++) {
            aug[i][j] = A[i][j];
            aug[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (size_t i = 0; i < n; i++) {
        double pivot = aug[i][i];
        if (fabs(pivot) < 1e-12) {
            for (size_t k = 0; k < n; k++) {
                free(aug[k]);
            }
            free(aug);
            return;
        }
        for (size_t j = 0; j < 2 * n; j++) {
            aug[i][j] /= pivot;
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