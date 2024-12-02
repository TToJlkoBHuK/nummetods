#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "utils.h"

void generate_matrix(double** A, size_t n, double cond_number, int is_spd) {
    double** U = (double**)malloc(n * sizeof(double*));
    double** V = (double**)malloc(n * sizeof(double*));
    double* S = (double*)malloc(n * sizeof(double));

    for (size_t i = 0; i < n; i++) {
        U[i] = (double*)malloc(n * sizeof(double));
        V[i] = (double*)malloc(n * sizeof(double));
    }

    random_orthogonal_matrix(U, n);
    random_orthogonal_matrix(V, n);

    S[0] = 1.0;
    S[n - 1] = 1.0 / cond_number;
    for (size_t i = 1; i < n - 1; i++) {
        S[i] = S[n - 1] + (S[0] - S[n - 1]) * ((double)rand() / RAND_MAX);
    }

    // A = U * S * V^T
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A[i][j] = 0.0;
            for (size_t k = 0; k < n; k++) {
                A[i][j] += U[i][k] * S[k] * V[j][k];
            }
        }
    }

    if (is_spd) {
        // A = A^T * A
        double** AtA = allocate_matrix(n);
        transpose_and_multiply(A, AtA, n);
        copy_matrix(AtA, A, n);
        free_matrix(AtA, n);
    }

    for (size_t i = 0; i < n; i++) {
        free(U[i]);
        free(V[i]);
    }
    free(U);
    free(V);
    free(S);
}

void perturb_matrix(double** A, size_t n, double perturbation, int perturb_max) {
    size_t index_i = 0, index_j = 0;
    double target_value = A[0][0];

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if ((perturb_max && fabs(A[i][j]) > fabs(target_value)) ||
                (!perturb_max && fabs(A[i][j]) < fabs(target_value))) {
                target_value = A[i][j];
                index_i = i;
                index_j = j;
            }
        }
    }

    A[index_i][index_j] *= (1.0 + perturbation * ((double)rand() / RAND_MAX));
}

void multiply_matrix_vector(double** A, double* x, double* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        b[i] = 0.0;
        for (size_t j = 0; j < n; j++) {
            b[i] += A[i][j] * x[j];
        }
    }
}

double calculate_condition_number(double** A, size_t n) {
    double** A_inv = allocate_matrix(n);
    invert_matrix(A, A_inv, n);
    double norm_A = matrix_infinity_norm(A, n);
    double norm_A_inv = matrix_infinity_norm(A_inv, n);
    free_matrix(A_inv, n);
    return norm_A * norm_A_inv;
}

void copy_matrix(double** src, double** dest, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            dest[i][j] = src[i][j];
        }
    }
}