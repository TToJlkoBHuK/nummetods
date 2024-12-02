#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "jacobi.h"
#include "matrix.h"
#include "utils.h"

#define N 10
#define MAX_ITER 10000

int main() {
    srand((unsigned int)time(NULL));

    double cond_numbers[] = { 1e0, 1e2, 1e4, 1e6, 1e8 };
    int num_cond = sizeof(cond_numbers) / sizeof(cond_numbers[0]);

    double tolerances[] = { 1e-15, 1e-14, 1e-13, 1e-12, 1e-11,
                           1e-10, 1e-9, 1e-8, 1e-7, 1e-6,
                           1e-5, 1e-4, 1e-3, 1e-2 };
    int num_tol = sizeof(tolerances) / sizeof(tolerances[0]);

    double** A = allocate_matrix(N);
    double* x_exact = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* x = (double*)malloc(N * sizeof(double));
    double* errors = (double*)malloc(MAX_ITER * sizeof(double));

    for (size_t i = 0; i < N; i++) {
        x_exact[i] = (double)(i + 1);
    }

    FILE* file_accuracy = fopen("accuracy_vs_cond.txt", "w");
    FILE* file_time = fopen("time_vs_cond.txt", "w");
    FILE* file_iterations = fopen("iterations_vs_tol.txt", "w");
    FILE* file_error_progress_good = fopen("error_progress_good.txt", "w");
    FILE* file_error_progress_bad = fopen("error_progress_bad.txt", "w");

    for (int idx = 0; idx < num_cond; idx++) {
        double cond_number = cond_numbers[idx];

        generate_matrix(A, N, cond_number, 0, 0); 

        double real_cond = calculate_condition_number(A, N);

        multiply_matrix_vector(A, x_exact, b, N);

        clock_t start_time = clock();
        int iterations = jacobi_method(A, b, x, N, 1e-10, MAX_ITER, errors);
        clock_t end_time = clock();

        double error = vector_norm_diff(x, x_exact, N);

        fprintf(file_accuracy, "%e %e\n", real_cond, error);
        fprintf(file_time, "%e %lf\n", real_cond, (double)(end_time - start_time) / CLOCKS_PER_SEC);
    }

    generate_matrix(A, N, 1e2, 0, 0);
    multiply_matrix_vector(A, x_exact, b, N);
    for (int i = 0; i < num_tol; i++) {
        double tol = tolerances[i];
        int iterations = jacobi_method(A, b, x, N, tol, MAX_ITER, errors);
        double error = vector_norm_diff(x, x_exact, N);
        fprintf(file_iterations, "%e %d %e\n", tol, iterations, error);
    }

    generate_matrix(A, N, 1e2, 0, 0);
    multiply_matrix_vector(A, x_exact, b, N);
    int iterations = jacobi_method(A, b, x, N, 1e-15, MAX_ITER, errors);
    FILE* file_error_good = fopen("error_progress_good.txt", "w");
    for (int i = 0; i < iterations; i++) {
        fprintf(file_error_good, "%d %e\n", i + 1, errors[i]);
    }
    fclose(file_error_good);


    generate_matrix(A, N, 1e8, 0, 0);
    multiply_matrix_vector(A, x_exact, b, N);
    iterations = jacobi_method(A, b, x, N, 1e-15, MAX_ITER, errors);
    FILE* file_error_bad = fopen("error_progress_bad.txt", "w");
    for (int i = 0; i < iterations; i++) {
        fprintf(file_error_bad, "%d %e\n", i + 1, errors[i]);
    }
    fclose(file_error_bad);

    double determinants[] = { 1e-15, 1e-14, 1e-13, 1e-12, 1e-11,
                             1e-10, 1e-9, 1e-8, 1e-7, 1e-6,
                             1e-5, 1e-4, 1e-3, 1e-2 };
    int num_det = sizeof(determinants) / sizeof(determinants[0]);

    FILE* file_error_determinant = fopen("error_vs_determinant.txt", "w");

    for (int idx = 0; idx < num_det; idx++) {
        double det_value = determinants[idx];
        generate_matrix(A, N, 1e2, 0, det_value);
        multiply_matrix_vector(A, x_exact, b, N);
        int iterations = jacobi_method(A, b, x, N, 1e-10, MAX_ITER, errors);
        double error = vector_norm_diff(x, x_exact, N);
        fprintf(file_error_determinant, "%e %e\n", det_value, error);
    }
    fclose(file_error_determinant);

    free_matrix(A, N);
    free(x_exact);
    free(b);
    free(x);
    free(errors);
    fclose(file_accuracy);
    fclose(file_time);
    fclose(file_iterations);

    return 0;
}