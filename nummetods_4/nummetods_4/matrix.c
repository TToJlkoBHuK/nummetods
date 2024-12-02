#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "utils.h"

void generate_matrix(double** A, size_t n, double cond_number, int is_spd, int determinant_case) {

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (i == j) {
                A[i][j] = 1.0;
            }
            else {
                A[i][j] = ((double)rand() / RAND_MAX) * 0.01;
            }
        }
    }

    A[0][0] = 1.0;
    A[n - 1][n - 1] = 1.0 / cond_number;

    if (is_spd) {
        // A = A^T * A
        double** At = allocate_matrix(n);
        transpose_matrix(A, At, n);

        double** AtA = allocate_matrix(n);
        multiply_matrices(At, A, AtA, n);

        copy_matrix(AtA, A, n);

        free_matrix(At, n);
        free_matrix(AtA, n);
    }

    if (determinant_case) {
        for (size_t i = 0; i < n; i++) {
            A[i][i] = determinant_case;
        }
    }
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
    double norm_A = matrix_norm(A, n);
    double norm_A_inv = matrix_norm(A_inv, n);
    free_matrix(A_inv, n);
    return norm_A * norm_A_inv;
}

double calculate_determinant(double** A, size_t n) {
    double det = 1.0;
    double** LU = allocate_matrix(n);
    copy_matrix(A, LU, n);

    for (size_t k = 0; k < n; k++) {
        if (LU[k][k] == 0) return 0.0;
        for (size_t i = k + 1; i < n; i++) {
            LU[i][k] /= LU[k][k];
            for (size_t j = k + 1; j < n; j++) {
                LU[i][j] -= LU[i][k] * LU[k][j];
            }
        }
        det *= LU[k][k];
    }

    free_matrix(LU, n);
    return det;
}