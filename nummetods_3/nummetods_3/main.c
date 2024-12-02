#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"
#include "givens.h"
#include "utils.h"

#define N 10
#define NUM_COND 8

int main() {
    srand((unsigned int)time(NULL));

    double cond_numbers[NUM_COND] = { 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8 };
    double perturbations[5] = { 0.01, 0.02, 0.03, 0.04, 0.05 };

    double** A = allocate_matrix(N);
    double* x_exact = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* x = (double*)malloc(N * sizeof(double));

    for (size_t i = 0; i < N; i++) {
        x_exact[i] = (double)(i + 1);
    }

    FILE* file_accuracy = fopen("accuracy_vs_cond.txt", "w");
    FILE* file_time = fopen("time_vs_cond.txt", "w");
    FILE* file_perturb_b = fopen("error_vs_perturb_b.txt", "w");
    FILE* file_perturb_A = fopen("error_vs_perturb_A.txt", "w");

    for (int idx = 0; idx < NUM_COND; idx++) {
        double cond_number = cond_numbers[idx];

        generate_matrix(A, N, cond_number, 0);

        double real_cond = calculate_condition_number(A, N);

        multiply_matrix_vector(A, x_exact, b, N);

        double** A_copy = allocate_matrix(N);
        copy_matrix(A, A_copy, N);
        double* b_copy = (double*)malloc(N * sizeof(double));
        for (size_t i = 0; i < N; i++) {
            b_copy[i] = b[i];
        }

        clock_t start_time = clock();
        givens_rotation(A_copy, b_copy, N);
        back_substitution(A_copy, b_copy, x, N);
        clock_t end_time = clock();

        double error = vector_infinity_norm(x, x_exact, N);

        fprintf(file_accuracy, "%e %e\n", real_cond, error);
        fprintf(file_time, "%e %lf\n", real_cond, (double)(end_time - start_time) / CLOCKS_PER_SEC);

        free_matrix(A_copy, N);
        free(b_copy);
    }

    double cond_number = 1e5;
    generate_matrix(A, N, cond_number, 0);
    multiply_matrix_vector(A, x_exact, b, N);

    double** A_copy = allocate_matrix(N);
    copy_matrix(A, A_copy, N);
    double* b_copy = (double*)malloc(N * sizeof(double));
    for (size_t i = 0; i < N; i++) {
        b_copy[i] = b[i];
    }

    givens_rotation(A_copy, b_copy, N);
    back_substitution(A_copy, b_copy, x, N);
    double error_base = vector_infinity_norm(x, x_exact, N);

    free_matrix(A_copy, N);
    free(b_copy);

    for (size_t idx = 0; idx < 5; idx++) {
        double perturbation = perturbations[idx];
        double* b_perturbed = (double*)malloc(N * sizeof(double));
        for (size_t i = 0; i < N; i++) {
            b_perturbed[i] = b[i] * (1.0 + ((double)rand() / RAND_MAX - 0.5) * 2 * perturbation);
        }

        double** A_copy = allocate_matrix(N);
        copy_matrix(A, A_copy, N);
        givens_rotation(A_copy, b_perturbed, N);
        back_substitution(A_copy, b_perturbed, x, N);
        double error = vector_infinity_norm(x, x_exact, N);
        double relative_error = error / error_base;
        fprintf(file_perturb_b, "%f %e\n", perturbation * 100, relative_error);

        free_matrix(A_copy, N);
        free(b_perturbed);
    }

    for (size_t idx = 0; idx < 5; idx++) {
        double perturbation = perturbations[idx];
        generate_matrix(A, N, cond_number, 0);
        perturb_matrix(A, N, perturbation, 1); 
        multiply_matrix_vector(A, x_exact, b, N);

        double** A_copy = allocate_matrix(N);
        copy_matrix(A, A_copy, N);
        double* b_copy = (double*)malloc(N * sizeof(double));
        for (size_t i = 0; i < N; i++) {
            b_copy[i] = b[i];
        }

        givens_rotation(A_copy, b_copy, N);
        back_substitution(A_copy, b_copy, x, N);
        double error = vector_infinity_norm(x, x_exact, N);
        double relative_error = error / error_base;
        fprintf(file_perturb_A, "%f %e\n", perturbation * 100, relative_error);

        free_matrix(A_copy, N);
        free(b_copy);
    }


    free_matrix(A, N);
    free(x_exact);
    free(b);
    free(x);
    fclose(file_accuracy);
    fclose(file_time);
    fclose(file_perturb_b);
    fclose(file_perturb_A);

    return 0;
}