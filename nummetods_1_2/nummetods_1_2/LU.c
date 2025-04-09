#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

#define N_MIN 10         
#define NUM_COND_STEPS 9 
#define NUM_PERTURB_REPEATS 20 
#define MAX_COND_NUM 1e8
#define NUM_TIMING_REPEATS 1000

typedef struct {
    double** data;
    int n;
} Matrix;

typedef struct {
    double* data;
    int n;
} Vector;

Matrix create_matrix(int n) {
    Matrix A;
    A.n = n;
    A.data = (double**)malloc(n * sizeof(double*));
    if (!A.data) return A;
    for (int i = 0; i < n; ++i) {
        A.data[i] = (double*)malloc(n * sizeof(double));
        if (!A.data[i]) {
            while (--i >= 0) free(A.data[i]);
            free(A.data);
            A.data = NULL;
            return A;
        }
    }
    return A;
}

void free_matrix(Matrix A) {
    if (A.data) {
        for (int i = 0; i < A.n; ++i) free(A.data[i]);
        free(A.data);
    }
}

Vector create_vector(int n) {
    Vector v;
    v.n = n;
    v.data = (double*)malloc(n * sizeof(double));
    return v;
}

void free_vector(Vector v) {
    free(v.data);
}

void print_matrix(Matrix A) {
    printf("Matrix (%dx%d):\n", A.n, A.n);
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.n; ++j) printf("%10.4f ", A.data[i][j]);
        printf("\n");
    }
}

void print_vector(Vector v) {
    printf("Vector (%d):\n", v.n);
    for (int i = 0; i < v.n; ++i) printf("%10.4f ", v.data[i]);
    printf("\n");
}

Matrix generate_matrix(int n, double target_cond) {
    Matrix A = create_matrix(n);
    if (!A.data) return A;

    Vector D_diag = create_vector(n);
    if (!D_diag.data) { free_matrix(A); A.data = NULL; return A; }

    D_diag.data[0] = target_cond;
    D_diag.data[n - 1] = 1.0;
    if (n > 1) {
        double ratio = pow(target_cond, 1.0 / (n - 1));
        for (int i = 1; i < n - 1; ++i) D_diag.data[i] = pow(ratio, n - 1 - i);
        for (int i = 0; i < n; ++i) {
            int j = rand() % n;
            double temp = D_diag.data[i];
            D_diag.data[i] = D_diag.data[j];
            D_diag.data[j] = temp;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) A.data[i][j] = D_diag.data[i];
            else A.data[i][j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }
    }

    free_vector(D_diag);
    return A;
}

Vector generate_exact_solution(int n) {
    Vector x = create_vector(n);
    for (int i = 0; i < n; ++i) x.data[i] = 1.0;
    return x;
}

int matrix_vector_mult(Matrix A, Vector x, Vector b) {
    if (A.n != x.n || A.n != b.n) return 0;
    for (int i = 0; i < A.n; ++i) {
        b.data[i] = 0.0;
        for (int j = 0; j < A.n; ++j) b.data[i] += A.data[i][j] * x.data[j];
    }
    return 1;
}

double infinity_norm(Vector v) {
    double max_val = 0.0;
    for (int i = 0; i < v.n; ++i) {
        double abs_val = fabs(v.data[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    return max_val;
}

int vector_subtract(Vector v1, Vector v2, Vector result) {
    if (v1.n != v2.n || v1.n != result.n) return 0;
    for (int i = 0; i < v1.n; ++i) result.data[i] = v1.data[i] - v2.data[i];
    return 1;
}

int lu_decomposition(Matrix A) {
    int n = A.n;
    for (int k = 0; k < n; ++k) {
        if (fabs(A.data[k][k]) < DBL_EPSILON) {
            fprintf(stderr, "Zero pivot at %d. Matrix is singular.\n", k);
            return 1;
        }
        for (int i = k + 1; i < n; ++i) {
            A.data[i][k] /= A.data[k][k];
            for (int j = k + 1; j < n; ++j)
                A.data[i][j] -= A.data[i][k] * A.data[k][j];
        }
    }
    return 0;
}

int forward_substitution(Matrix LU, Vector b, Vector y) {
    int n = LU.n;
    if (n != b.n || n != y.n) return 0;
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) sum += LU.data[i][j] * y.data[j];
        y.data[i] = b.data[i] - sum;
    }
    return 1;
}

int backward_substitution(Matrix LU, Vector y, Vector x) {
    int n = LU.n;
    if (n != y.n || n != x.n) return 0;
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) sum += LU.data[i][j] * x.data[j];
        if (fabs(LU.data[i][i]) < DBL_EPSILON * 100) {
            fprintf(stderr, "Division by near-zero in backward substitution at %d\n", i);
            for (int k = i; k >= 0; --k) x.data[k] = NAN;
            return 0;
        }
        x.data[i] = (y.data[i] - sum) / LU.data[i][i];
    }
    return 1;
}

void find_matrix_element(Matrix A, int* r, int* c, int find_min) {
    double best_val = find_min ? DBL_MAX : 0.0;
    *r = *c = 0;
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.n; ++j) {
            double current = fabs(A.data[i][j]);
            if ((find_min && current < best_val) || (!find_min && current > best_val)) {
                best_val = current;
                *r = i;
                *c = j;
            }
        }
    }
}

Matrix copy_matrix(Matrix A) {
    Matrix B = create_matrix(A.n);
    if (!B.data) return B;
    for (int i = 0; i < A.n; ++i)
        for (int j = 0; j < A.n; ++j) B.data[i][j] = A.data[i][j];
    return B;
}

Vector copy_vector(Vector v) {
    Vector u = create_vector(v.n);
    if (!u.data) return u;
    for (int i = 0; i < v.n; ++i) u.data[i] = v.data[i];
    return u;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    int n = N_MIN;
    if (argc > 1) n = atoi(argv[1]);
    if (n < 2) { fprintf(stderr, "Matrix size must be at least 2.\n"); return 1; }
    printf("Using matrix size N = %d\n", n);

    FILE* f_cond = fopen("lu_results_cond.txt", "w");
    FILE* f_pert_b = fopen("lu_results_pert_b.txt", "w");
    FILE* f_pert_a = fopen("lu_results_pert_A.txt", "w");
    if (!f_cond || !f_pert_b || !f_pert_a) { perror("Error opening files"); return 1; }

    fprintf(f_cond, "N TargetCond Time ErrorInf\n");
    fprintf(f_pert_b, "TargetCond Perturbation RelErrorAvg\n");
    fprintf(f_pert_a, "TargetCond Perturbation ElementType RelErrorAvg\n");

    for (int i = 0; i < NUM_COND_STEPS; ++i) {
        double target_cond = (i == 0) ? 1.0 : pow(10, i);
        if (target_cond > MAX_COND_NUM) target_cond = MAX_COND_NUM;

        printf("\n--- Testing for Target Condition Number: %.1e ---\n", target_cond);
        Matrix A = generate_matrix(n, target_cond);
        Vector x_exact = generate_exact_solution(n);
        Vector b = create_vector(n);
        matrix_vector_mult(A, x_exact, b);

        Matrix A_lu = copy_matrix(A);
        Vector x_computed = create_vector(n);
        Vector y = create_vector(n);

        clock_t start = clock();
        int lu_status = lu_decomposition(A_lu);
        if (lu_status == 0) {
            if (!forward_substitution(A_lu, b, y)) lu_status = -1;
            else if (!backward_substitution(A_lu, y, x_computed)) lu_status = -1;
        }
        clock_t end = clock();
        double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

        double error_inf = -1.0;
        if (lu_status == 0) {
            Vector error_vec = create_vector(n);
            if (error_vec.data) {
                vector_subtract(x_exact, x_computed, error_vec);
                error_inf = infinity_norm(error_vec);
                free_vector(error_vec);
            }
        }
        printf("Time: %f, Error: %g\n", cpu_time_used, error_inf);
        fprintf(f_cond, "%d %.1e %f %g\n", n, target_cond, cpu_time_used, error_inf);

        double norm_x_exact = infinity_norm(x_exact);
        if (norm_x_exact == 0) norm_x_exact = 1.0;

        // Perturb b
        for (int p = 1; p <= 5; ++p) {
            double perturbation = p / 100.0;
            double total_rel_error = 0;
            int successful = 0;
            for (int k = 0; k < NUM_PERTURB_REPEATS; ++k) {
                Vector b_pert = copy_vector(b);
                for (int j = 0; j < n; ++j)
                    b_pert.data[j] += perturbation * b.data[j] * (2.0 * rand() / RAND_MAX - 1.0);
                Matrix A_lu_copy = copy_matrix(A_lu);
                Vector y_pert = create_vector(n);
                Vector x_pert = create_vector(n);
                if (forward_substitution(A_lu_copy, b_pert, y_pert) && backward_substitution(A_lu_copy, y_pert, x_pert)) {
                    Vector err = create_vector(n);
                    vector_subtract(x_exact, x_pert, err);
                    total_rel_error += infinity_norm(err) / norm_x_exact;
                    successful++;
                    free_vector(err);
                }
                free_matrix(A_lu_copy);
                free_vector(y_pert);
                free_vector(x_pert);
                free_vector(b_pert);
            }
            double avg_error = (successful > 0) ? total_rel_error / successful : NAN;
            fprintf(f_pert_b, "%.1e %d %g\n", target_cond, p, avg_error);
        }

        // Perturb A
        int min_r, min_c, max_r, max_c;
        find_matrix_element(A, &min_r, &min_c, 1);
        find_matrix_element(A, &max_r, &max_c, 0);
        double min_val = A.data[min_r][min_c];
        double max_val = A.data[max_r][max_c];

        for (int p = 1; p <= 5; ++p) {
            double perturbation = p / 100.0;
            double total_min = 0, total_max = 0;
            int successful_min = 0, successful_max = 0;

            for (int k = 0; k < NUM_PERTURB_REPEATS; ++k) {
                // Perturb min element
                Matrix A_min = copy_matrix(A);
                A_min.data[min_r][min_c] += perturbation * min_val * (2.0 * rand() / RAND_MAX - 1.0);
                Matrix A_lu_min = copy_matrix(A_min);
                Vector y_min = create_vector(n);
                Vector x_min = create_vector(n);
                if (lu_decomposition(A_lu_min) == 0 && forward_substitution(A_lu_min, b, y_min) && backward_substitution(A_lu_min, y_min, x_min)) {
                    Vector err = create_vector(n);
                    vector_subtract(x_exact, x_min, err);
                    total_min += infinity_norm(err) / norm_x_exact;
                    successful_min++;
                    free_vector(err);
                }
                free_matrix(A_lu_min);
                free_matrix(A_min);
                free_vector(y_min);
                free_vector(x_min);

                // Perturb max element
                Matrix A_max = copy_matrix(A);
                A_max.data[max_r][max_c] += perturbation * max_val * (2.0 * rand() / RAND_MAX - 1.0);
                Matrix A_lu_max = copy_matrix(A_max);
                Vector y_max = create_vector(n);
                Vector x_max = create_vector(n);
                if (lu_decomposition(A_lu_max) == 0 && forward_substitution(A_lu_max, b, y_max) && backward_substitution(A_lu_max, y_max, x_max)) {
                    Vector err = create_vector(n);
                    vector_subtract(x_exact, x_max, err);
                    total_max += infinity_norm(err) / norm_x_exact;
                    successful_max++;
                    free_vector(err);
                }
                free_matrix(A_lu_max);
                free_matrix(A_max);
                free_vector(y_max);
                free_vector(x_max);
            }

            double avg_min = (successful_min > 0) ? total_min / successful_min : NAN;
            double avg_max = (successful_max > 0) ? total_max / successful_max : NAN;
            fprintf(f_pert_a, "%.1e %d %d %g\n", target_cond, p, 0, avg_min);
            fprintf(f_pert_a, "%.1e %d %d %g\n", target_cond, p, 1, avg_max);
        }

        free_matrix(A);
        free_vector(x_exact);
        free_vector(b);
        free_matrix(A_lu);
        free_vector(x_computed);
        free_vector(y);
    }

    fclose(f_cond);
    fclose(f_pert_b);
    fclose(f_pert_a);
    printf("\nResults saved to files.\n");
    return 0;
}