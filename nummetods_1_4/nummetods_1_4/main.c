#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// --- Matrix/Vector Utilities ---

typedef struct {
    double* data;
    int n;
} Matrix;

typedef struct {
    double* data;
    int n;
} Vector;

// Allocate memory for a matrix
Matrix create_matrix(int n) {
    Matrix m;
    m.n = n;
    m.data = (double*)malloc(n * n * sizeof(double));
    if (m.data == NULL) {
        fprintf(stderr, "Error allocating matrix data\n");
        exit(EXIT_FAILURE);
    }
    return m;
}

// Free matrix memory
void free_matrix(Matrix* m) {
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
        m->n = 0;
    }
}

// Allocate memory for a vector
Vector create_vector(int n) {
    Vector v;
    v.n = n;
    v.data = (double*)malloc(n * sizeof(double));
    if (v.data == NULL) {
        fprintf(stderr, "Error allocating vector data\n");
        exit(EXIT_FAILURE);
    }
    return v;
}

// Free vector memory
void free_vector(Vector* v) {
    if (v && v->data) {
        free(v->data);
        v->data = NULL;
        v->n = 0;
    }
}

// Matrix-vector multiplication: y = A * x
void matrix_vector_mult(const Matrix* A, const Vector* x, Vector* y) {
    int n = A->n;
    for (int i = 0; i < n; ++i) {
        y->data[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            y->data[i] += A->data[i * n + j] * x->data[j];
        }
    }
}

// Vector subtraction: result = a - b
void vector_subtract(const Vector* a, const Vector* b, Vector* result) {
    int n = a->n;
    for (int i = 0; i < n; ++i) {
        result->data[i] = a->data[i] - b->data[i];
    }
}

// Vector addition: result = a + b
void vector_add(const Vector* a, const Vector* b, Vector* result) {
    int n = a->n;
    for (int i = 0; i < n; ++i) {
        result->data[i] = a->data[i] + b->data[i];
    }
}


// Scale vector: result = scalar * v
void vector_scale(double scalar, const Vector* v, Vector* result) {
    int n = v->n;
    for (int i = 0; i < n; ++i) {
        result->data[i] = scalar * v->data[i];
    }
}

// Calculate infinity norm of a vector
double infinity_norm(const Vector* v) {
    double max_val = 0.0;
    int n = v->n;
    for (int i = 0; i < n; ++i) {
        double abs_val = fabs(v->data[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    return max_val;
}

// Calculate infinity norm of a matrix
double matrix_infinity_norm(const Matrix* A) {
    int n = A->n;
    double max_row_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double current_row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            current_row_sum += fabs(A->data[i * n + j]);
        }
        if (current_row_sum > max_row_sum) {
            max_row_sum = current_row_sum;
        }
    }
    return max_row_sum;
}


// Fill vector with zeros
void zero_vector(Vector* v) {
    int n = v->n;
    for (int i = 0; i < n; ++i) {
        v->data[i] = 0.0;
    }
}

// Copy vector src to dest
void copy_vector(const Vector* src, Vector* dest) {
    if (src->n != dest->n) {
        fprintf(stderr, "Error: Cannot copy vectors of different sizes.\n");
        return;
    }
    memcpy(dest->data, src->data, src->n * sizeof(double));
}


// --- Matrix Generation ---

// Generate a random double between min and max
double rand_double(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

Matrix generate_matrix_with_cond(int n, double condition_number) {
    Matrix A = create_matrix(n);
    Matrix D = create_matrix(n); // Diagonal matrix
    Matrix P = create_matrix(n); // Orthogonal matrix P
    Matrix Q = create_matrix(n); // Orthogonal matrix Q
    Matrix Pt = create_matrix(n); // Transpose of P
    Matrix Temp = create_matrix(n); // Temporary storage

    // Initialize P and Q to identity
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            P.data[i * n + j] = (i == j) ? 1.0 : 0.0;
            Q.data[i * n + j] = (i == j) ? 1.0 : 0.0;
            D.data[i * n + j] = 0.0;
        }
    }

    double log_cond = log(condition_number);
    double min_log_s = -log_cond / 2.0;
    double max_log_s = log_cond / 2.0;

    for (int i = 0; i < n; ++i) {
        double log_s = min_log_s + (max_log_s - min_log_s) * ((double)i / (n - 1));
        D.data[i * n + i] = exp(log_s);
        if (D.data[i * n + i] < 1e-10) D.data[i * n + i] = 1e-10; // Avoid too small values
    }
    D.data[0] = sqrt(condition_number);
    D.data[(n - 1) * n + (n - 1)] = 1.0 / sqrt(condition_number);

    for (int k = 0; k < n * n; ++k) {
        int i = rand() % n;
        int j = rand() % n;
        if (i >= j) j = (i + 1 + rand() % (n - 1)) % n; // Ensure i != j

        double angle = rand_double(0, 2.0 * M_PI);
        double c = cos(angle);
        double s = sin(angle);

        // Rotate P
        for (int row = 0; row < n; ++row) {
            double p_ri = P.data[row * n + i];
            double p_rj = P.data[row * n + j];
            P.data[row * n + i] = c * p_ri - s * p_rj;
            P.data[row * n + j] = s * p_ri + c * p_rj;
        }

        for (int col = 0; col < n; ++col) {
            double q_ic = Q.data[i * n + col];
            double q_jc = Q.data[j * n + col];
            Q.data[i * n + col] = c * q_ic - s * q_jc;
            Q.data[j * n + col] = s * q_ic + c * q_jc;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Temp.data[i * n + j] = P.data[i * n + j] * D.data[j * n + j];
        }
    }

    // A = Temp * Q
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += Temp.data[i * n + k] * Q.data[k * n + j];
            }
            A.data[i * n + j] = sum;
        }
    }


    free_matrix(&D);
    free_matrix(&P);
    free_matrix(&Q);
    free_matrix(&Pt);
    free_matrix(&Temp);

    return A;
}

int solve_msi(const Matrix* A, const Vector* b, Vector* x, double tau,
    int max_iterations, double tolerance,
    double* final_residual_norm, int log_history, FILE* history_file)
{
    int n = A->n;
    Vector Ax = create_vector(n);
    Vector residual = create_vector(n);
    Vector delta_x = create_vector(n);
    Vector x_prev = create_vector(n);

    int iterations = 0;
    double current_norm = 0.0;

    if (log_history && history_file) {
        matrix_vector_mult(A, x, &Ax);
        vector_subtract(b, &Ax, &residual);
        vector_scale(tau, &residual, &delta_x);
        current_norm = infinity_norm(&delta_x);
        fprintf(history_file, "%d %.16e\n", 0, current_norm);
    }

    for (iterations = 0; iterations < max_iterations; iterations++) {
        copy_vector(x, &x_prev);
        matrix_vector_mult(A, &x_prev, &Ax);
        vector_subtract(b, &Ax, &residual);
        vector_scale(tau, &residual, &delta_x);
        current_norm = infinity_norm(&delta_x);
        vector_add(&x_prev, &delta_x, x);

        if (log_history && history_file) {
            fprintf(history_file, "%d %.16e\n", iterations + 1, current_norm);
        }

        if (current_norm < tolerance) {
            iterations++;
            break;
        }

        if (isnan(current_norm) || isinf(current_norm)) {
            fprintf(stderr, "Warning: MSI diverged after %d iterations.\n", iterations);
            break;
        }
    }

    matrix_vector_mult(A, x, &Ax);
    vector_subtract(b, &Ax, &residual);
    *final_residual_norm = infinity_norm(&residual);

    free_vector(&Ax);
    free_vector(&residual);
    free_vector(&delta_x);
    free_vector(&x_prev);

    return iterations;
}


// --- Main Experiment Loop ---

int main() {
    srand(time(NULL));
    int n = 20;

    double condition_numbers[] = { 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8 };
    int num_conds = sizeof(condition_numbers) / sizeof(double);

    double tolerances[] = { 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15 };
    int num_tols = sizeof(tolerances) / sizeof(double);

    int max_iterations = 50000;

    FILE* results_file = fopen("results.txt", "w");
    if (!results_file) {
        perror("Error opening results.txt");
        return EXIT_FAILURE;
    }
    fprintf(results_file, "ConditionNumber TargetTolerance Iterations Time ActualError FinalResidualNorm\n");

    Vector x_exact = create_vector(n);
    for (int i = 0; i < n; ++i) {
        x_exact.data[i] = (double)(i + 1);
    }

    for (int c_idx = 0; c_idx < num_conds; ++c_idx) {
        double current_cond = condition_numbers[c_idx];
        printf("Processing Condition Number: %.2e\n", current_cond);

        Matrix A = generate_matrix_with_cond(n, current_cond);
        Vector b = create_vector(n);
        matrix_vector_mult(&A, &x_exact, &b);

        double a_norm_inf = matrix_infinity_norm(&A);
        double tau = 1.0 / a_norm_inf;
        if (current_cond > 1e6) {
            tau *= 0.5;
        }
        printf("  Using tau = %.4e (||A||_inf = %.4e)\n", tau, a_norm_inf);

        for (int t_idx = 0; t_idx < num_tols; ++t_idx) {
            double current_tol = tolerances[t_idx];
            Vector x = create_vector(n);
            zero_vector(&x);

            double final_residual_norm;
            int iterations;
            double elapsed_time;

            clock_t start = clock();
            iterations = solve_msi(&A, &b, &x, tau, max_iterations, current_tol, &final_residual_norm, 0, NULL);
            clock_t end = clock();
            elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

            Vector error_vec = create_vector(n);
            vector_subtract(&x, &x_exact, &error_vec);
            double actual_error = infinity_norm(&error_vec);

            printf("  Tol: %.1e, Iters: %d, Time: %.4f s, ActualErr: %.4e, ResNorm: %.4e\n",
                current_tol, iterations, elapsed_time, actual_error, final_residual_norm);

            if (iterations >= max_iterations && actual_error > current_tol * 100 && actual_error > 1e-1) {
                fprintf(stderr, "    Warning: Possible non-convergence for cond=%.1e, tol=%.1e\n", current_cond, current_tol);
                actual_error = 1.0;
                final_residual_norm = 1.0;
                iterations = max_iterations;
            }

            fprintf(results_file, "%.8e %.16e %d %.8f %.16e %.16e\n",
                current_cond, current_tol, iterations, elapsed_time, actual_error, final_residual_norm);

            free_vector(&x);
            free_vector(&error_vec);
        }

        if (fabs(current_cond - 10.0) < 1.0 || fabs(current_cond - 1e6) < 1.0) {
            char history_filename[100];
            sprintf(history_filename, "history_cond_%.0e.txt", current_cond);
            FILE* history_file = fopen(history_filename, "w");
            if (history_file) {
                printf("  Logging iteration history to %s\n", history_filename);
                Vector x_hist = create_vector(n);
                zero_vector(&x_hist);
                double final_res_norm_hist;

                solve_msi(&A, &b, &x_hist, tau, max_iterations, 1e-10, &final_res_norm_hist, 1, history_file);
                fclose(history_file);
                free_vector(&x_hist);
            }
            else {
                fprintf(stderr, "Error opening history file %s\n", history_filename);
            }
        }

        free_matrix(&A);
        free_vector(&b);
    }

    free_vector(&x_exact);
    fclose(results_file);

    printf("\nExperiments finished. Results saved to results.txt\n");
    printf("History files (history_cond_*.txt) generated for selected condition numbers.\n");

    return EXIT_SUCCESS;
}