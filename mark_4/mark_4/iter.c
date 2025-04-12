#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // For memcpy

#define MAX_ITER 100000 // Max iterations before giving up
#define RND_SCALE 0.01  // Scale for random noise during matrix generation

// Structure to hold iteration results for plotting error vs iteration
typedef struct {
    int iter_count;
    double* errors; // Stores ||x_k - x_exact||_inf at each iteration
} IterationHistory;

// --- Memory Allocation ---
double** allocate_matrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    if (!matrix) return NULL;
    matrix[0] = (double*)malloc(n * n * sizeof(double));
    if (!matrix[0]) {
        free(matrix);
        return NULL;
    }
    for (int i = 1; i < n; ++i) {
        matrix[i] = matrix[0] + i * n;
    }
    return matrix;
}

void free_matrix(double** matrix) {
    if (matrix) {
        if (matrix[0]) free(matrix[0]);
        free(matrix);
    }
}

double* allocate_vector(int n) {
    double* vector = (double*)malloc(n * sizeof(double));
    return vector;
}

void free_vector(double* vector) {
    if (vector) free(vector);
}

// --- Vector/Matrix Operations ---

// y = A * x
void matrix_vector_mult(int n, double** A, double* x, double* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            y[i] += A[i][j] * x[j];
        }
    }
}

// result = v1 - v2
void vector_sub(int n, double* v1, double* v2, double* result) {
    for (int i = 0; i < n; ++i) {
        result[i] = v1[i] - v2[i];
    }
}

// result = v1 + v2
void vector_add(int n, double* v1, double* v2, double* result) {
    for (int i = 0; i < n; ++i) {
        result[i] = v1[i] + v2[i];
    }
}

// result = scalar * v
void scalar_vector_mult(int n, double scalar, double* v, double* result) {
    for (int i = 0; i < n; ++i) {
        result[i] = scalar * v[i];
    }
}

// Infinity norm: max|v_i|
double vector_norm_inf(int n, double* v) {
    double max_val = 0.0;
    for (int i = 0; i < n; ++i) {
        if (fabs(v[i]) > max_val) {
            max_val = fabs(v[i]);
        }
    }
    return max_val;
}

// --- Matrix Generation ---
// Creates a matrix A with approximate condition number 'cond'.
// Note: This is a simplified generation. Real control requires SVD.
// It creates a diagonal matrix D and adds noise.
// For the non-SPD requirement, this noise makes it non-symmetric generally.
void generate_matrix(int n, double cond, double** A) {
    // 1. Create diagonal matrix D with desired eigenvalues spread
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = 0.0;
        }
        // Logarithmic distribution of eigenvalues between 1 and cond
        if (cond == 1.0) {
            A[i][i] = 1.0;
        }
        else {
            A[i][i] = pow(cond, (double)i / (n - 1));
            // Alternative: Linear distribution
            // A[i][i] = 1.0 + (cond - 1.0) * ((double)i / (n - 1.0));
        }
    }
    // Ensure min eigenvalue is 1.0 and max is cond
    if (n > 1) {
        A[0][0] = 1.0;
        A[n - 1][n - 1] = cond;
    }


    // 2. Add small random noise to make it non-diagonal/non-symmetric
    // Ensures violation of SPD property for general case.
    // Use a fixed seed for reproducibility during tests if needed
    // srand(time(NULL)); // Or srand(some_fixed_seed);
    double max_diag = cond > 1.0 ? cond : 1.0; // Use max eigenvalue for scaling noise
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // Add noise scaled by max diagonal element and RND_SCALE
            // Make noise smaller for off-diagonal to maintain some diagonal dominance
            double noise_scale = (i == j) ? RND_SCALE * 0.1 : RND_SCALE;
            A[i][j] += noise_scale * max_diag * ((double)rand() / RAND_MAX - 0.5) * 2.0;
        }
    }
}

// --- Simple Iteration Method ---
// Solves Ax = b using x_k+1 = x_k + tau * (b - A*x_k)
// Stores error history if history != NULL
int simple_iteration(int n, double** A, double* b, double* x_sol,
    double cond, double tolerance, int max_iter,
    double* x_exact, IterationHistory* history) {

    double* x_k = allocate_vector(n);      // Current iteration x
    double* x_k1 = allocate_vector(n);     // Next iteration x
    double* residual = allocate_vector(n); // b - A*x_k
    double* Ax = allocate_vector(n);       // Stores A*x_k
    double* temp_vec = allocate_vector(n); // For intermediate calcs like tau*residual

    if (!x_k || !x_k1 || !residual || !Ax || !temp_vec) {
        fprintf(stderr, "Error allocating vectors in simple_iteration\n");
        free_vector(x_k); free_vector(x_k1); free_vector(residual);
        free_vector(Ax); free_vector(temp_vec);
        return -1; // Error code
    }

    // Initial guess: zero vector
    for (int i = 0; i < n; ++i) x_k[i] = 0.0;

    double max_abs_aii = 0.0;
    for (int i = 0; i < n; ++i) {
        if (fabs(A[i][i]) > max_abs_aii) {
            max_abs_aii = fabs(A[i][i]);
        }
    }
    if (max_abs_aii < 1e-9) {
        max_abs_aii = 1.0;
    }
    double tau = 1.0 / max_abs_aii;

    if (history) {
        history->errors = (double*)malloc(max_iter * sizeof(double));
        if (!history->errors) {
            fprintf(stderr, "Error allocating history buffer\n");
            history = NULL; // Cannot store history
        }
        history->iter_count = 0;
    }


    int k = 0;
    for (k = 0; k < max_iter; ++k) {
        // Calculate residual: r = b - A * x_k
        matrix_vector_mult(n, A, x_k, Ax);
        vector_sub(n, b, Ax, residual);

        // Check stopping criterion: ||residual||_inf < tolerance
        double residual_norm = vector_norm_inf(n, residual);
        if (residual_norm < tolerance) {
            break; // Converged
        }

        // Store error ||x_k - x_exact||_inf for history
        if (history && history->errors) {
            vector_sub(n, x_k, x_exact, temp_vec); // temp_vec = x_k - x_exact
            history->errors[k] = vector_norm_inf(n, temp_vec);
            history->iter_count++;
        }

        // Update x: x_{k+1} = x_k + tau * residual
        scalar_vector_mult(n, tau, residual, temp_vec); // temp_vec = tau * residual
        vector_add(n, x_k, temp_vec, x_k1);          // x_k1 = x_k + temp_vec

        // Prepare for next iteration: x_k = x_k1
        memcpy(x_k, x_k1, n * sizeof(double));
    }

    // Store final solution
    memcpy(x_sol, x_k, n * sizeof(double));

    // Store final error if history requested and loop finished before max_iter
    if (history && history->errors && k < max_iter) {
        vector_sub(n, x_k, x_exact, temp_vec);
        history->errors[k] = vector_norm_inf(n, temp_vec);
        history->iter_count++;
    }


    free_vector(x_k);
    free_vector(x_k1);
    free_vector(residual);
    free_vector(Ax);
    free_vector(temp_vec);

    if (k == max_iter) {
        //fprintf(stderr, "Warning: Simple Iteration did not converge within %d iterations for cond=%.1e, tol=%.1e\n", max_iter, cond, tolerance);
        return max_iter; // Return max_iter to indicate non-convergence within limit
    }

    return k; // Return number of iterations performed
}


int main() {
    int n = 10; // Matrix dimension (>= 10)
    srand(time(NULL)); // Seed random number generator

    // --- Define Test Parameters ---
    double conds[] = { 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8 };
    int num_conds = sizeof(conds) / sizeof(conds[0]);

    double epsilons[] = { 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15 };
    // Reverse order for easier plotting later if needed, but can handle in python too
    int num_eps = sizeof(epsilons) / sizeof(epsilons[0]);

    // --- Allocate Memory ---
    double** A = allocate_matrix(n);
    double* b = allocate_vector(n);
    double* x_exact = allocate_vector(n);
    double* x_sol = allocate_vector(n);
    double* error_vec = allocate_vector(n); // x_sol - x_exact

    if (!A || !b || !x_exact || !x_sol || !error_vec) {
        fprintf(stderr, "Error allocating main matrices/vectors\n");
        free_matrix(A); free_vector(b); free_vector(x_exact);
        free_vector(x_sol); free_vector(error_vec);
        return 1;
    }

    // --- Define Exact Solution ---
    for (int i = 0; i < n; ++i) {
        x_exact[i] = (double)(i + 1); // e.g., [1, 2, ..., n]
    }

    // --- Prepare Output Files ---
    FILE* f_acc_cond = fopen("results_accuracy_vs_cond.txt", "w");
    FILE* f_time_cond = fopen("results_time_vs_cond.txt", "w");
    FILE* f_err_eps_good = fopen("results_error_vs_eps_good.txt", "w");
    FILE* f_err_eps_bad = fopen("results_error_vs_eps_bad.txt", "w");
    FILE* f_iter_eps_good = fopen("results_iters_vs_eps_good.txt", "w");
    FILE* f_iter_eps_bad = fopen("results_iters_vs_eps_bad.txt", "w");
    FILE* f_err_iter_good = fopen("results_error_vs_iter_good.txt", "w");
    FILE* f_err_iter_bad = fopen("results_error_vs_iter_bad.txt", "w");

    if (!f_acc_cond || !f_time_cond || !f_err_eps_good || !f_err_eps_bad ||
        !f_iter_eps_good || !f_iter_eps_bad || !f_err_iter_good || !f_err_iter_bad) {
        fprintf(stderr, "Error opening output files!\n");
        // Clean up allocated memory
        free_matrix(A); free_vector(b); free_vector(x_exact);
        free_vector(x_sol); free_vector(error_vec);
        // Close any opened files (optional, OS usually handles)
        if (f_acc_cond) fclose(f_acc_cond);
        if (f_time_cond) fclose(f_time_cond);
        // ... close others
        return 1;
    }

    // Headers for files
    fprintf(f_acc_cond, "ConditionNumber FinalAccuracy(InfNorm)\n");
    fprintf(f_time_cond, "ConditionNumber ExecutionTime(s)\n");
    fprintf(f_err_eps_good, "TargetEpsilon FinalAccuracy(InfNorm)\n");
    fprintf(f_err_eps_bad, "TargetEpsilon FinalAccuracy(InfNorm)\n");
    fprintf(f_iter_eps_good, "TargetEpsilon Iterations\n");
    fprintf(f_iter_eps_bad, "TargetEpsilon Iterations\n");
    fprintf(f_err_iter_good, "Iteration Error(InfNorm)\n");
    fprintf(f_err_iter_bad, "Iteration Error(InfNorm)\n");

    // --- Experiment 1 & 2: Accuracy/Time vs Condition Number ---
    printf("Running Experiment 1 & 2: Accuracy/Time vs Condition Number...\n");
    double fixed_eps_for_exp12 = 1e-10; // Fixed precision for this experiment
    for (int i = 0; i < num_conds; ++i) {
        double current_cond = conds[i];
        printf("  Cond = %.1e\n", current_cond);

        generate_matrix(n, current_cond, A);
        matrix_vector_mult(n, A, x_exact, b); // b = A * x_exact

        clock_t start = clock();
        int iters = simple_iteration(n, A, b, x_sol, current_cond, fixed_eps_for_exp12, MAX_ITER, x_exact, NULL); // No history needed here
        clock_t end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

        vector_sub(n, x_sol, x_exact, error_vec);
        double final_accuracy = vector_norm_inf(n, error_vec);

        if (iters < MAX_ITER) { // Record only if converged
            fprintf(f_acc_cond, "%.10e %.10e\n", current_cond, final_accuracy);
            fprintf(f_time_cond, "%.10e %.10e\n", current_cond, time_taken);
        }
        else {
            printf("    Skipping data point for cond=%.1e (did not converge within %d iters)\n", current_cond, MAX_ITER);
            // Optional: write placeholder to file if needed
            fprintf(f_acc_cond, "%.10e NAN\n", current_cond);
            fprintf(f_time_cond, "%.10e NAN\n", current_cond);
        }
    }
    printf("Experiment 1 & 2 finished.\n");

    // --- Experiment 3 & 4: Error/Iterations vs Epsilon (Good/Bad Cond) ---
    printf("Running Experiment 3 & 4: Error/Iterations vs Epsilon...\n");
    double good_cond = 10.0; // Example of good conditioning
    double bad_cond = 1e5;   // Example of bad conditioning

    // Generate matrices once for this experiment
    double** A_good = allocate_matrix(n);
    double* b_good = allocate_vector(n);
    generate_matrix(n, good_cond, A_good);
    matrix_vector_mult(n, A_good, x_exact, b_good);

    double** A_bad = allocate_matrix(n);
    double* b_bad = allocate_vector(n);
    generate_matrix(n, bad_cond, A_bad);
    matrix_vector_mult(n, A_bad, x_exact, b_bad);

    for (int i = 0; i < num_eps; ++i) {
        double current_eps = epsilons[i];
        printf("  Epsilon = %.1e\n", current_eps);

        // Good conditioning
        int iters_good = simple_iteration(n, A_good, b_good, x_sol, good_cond, current_eps, MAX_ITER, x_exact, NULL);
        vector_sub(n, x_sol, x_exact, error_vec);
        double final_acc_good = vector_norm_inf(n, error_vec);
        if (iters_good < MAX_ITER) {
            fprintf(f_err_eps_good, "%.10e %.10e\n", current_eps, final_acc_good);
            fprintf(f_iter_eps_good, "%.10e %d\n", current_eps, iters_good);
        }
        else {
            fprintf(f_err_eps_good, "%.10e NAN\n", current_eps);
            fprintf(f_iter_eps_good, "%.10e -1\n", current_eps); // Indicate non-convergence
        }


        // Bad conditioning
        int iters_bad = simple_iteration(n, A_bad, b_bad, x_sol, bad_cond, current_eps, MAX_ITER, x_exact, NULL);
        vector_sub(n, x_sol, x_exact, error_vec);
        double final_acc_bad = vector_norm_inf(n, error_vec);
        if (iters_bad < MAX_ITER) {
            fprintf(f_err_eps_bad, "%.10e %.10e\n", current_eps, final_acc_bad);
            fprintf(f_iter_eps_bad, "%.10e %d\n", current_eps, iters_bad);
        }
        else {
            fprintf(f_err_eps_bad, "%.10e NAN\n", current_eps);
            fprintf(f_iter_eps_bad, "%.10e -1\n", current_eps); // Indicate non-convergence
        }
    }
    printf("Experiment 3 & 4 finished.\n");

    // --- Experiment 5: Error vs Iterations (Good/Bad Cond) ---
    printf("Running Experiment 5: Error vs Iterations...\n");
    double fixed_eps_for_exp5 = 1e-16; // Use a very small tolerance to run many iterations
    int max_iter_exp5 = 10000; // Limit iterations for history tracking

    IterationHistory hist_good = { 0, NULL };
    IterationHistory hist_bad = { 0, NULL };

    printf("  Good Cond (%.1e)...\n", good_cond);
    int iters_hist_good = simple_iteration(n, A_good, b_good, x_sol, good_cond, fixed_eps_for_exp5, max_iter_exp5, x_exact, &hist_good);
    if (hist_good.errors) {
        for (int k = 0; k < hist_good.iter_count; ++k) {
            fprintf(f_err_iter_good, "%d %.10e\n", k, hist_good.errors[k]);
        }
        free(hist_good.errors); // Free history buffer
    }

    printf("  Bad Cond (%.1e)...\n", bad_cond);
    int iters_hist_bad = simple_iteration(n, A_bad, b_bad, x_sol, bad_cond, fixed_eps_for_exp5, max_iter_exp5, x_exact, &hist_bad);
    if (hist_bad.errors) {
        for (int k = 0; k < hist_bad.iter_count; ++k) {
            fprintf(f_err_iter_bad, "%d %.10e\n", k, hist_bad.errors[k]);
        }
        free(hist_bad.errors); // Free history buffer
    }
    printf("Experiment 5 finished.\n");


    // --- Cleanup ---
    printf("Cleaning up...\n");
    fclose(f_acc_cond);
    fclose(f_time_cond);
    fclose(f_err_eps_good);
    fclose(f_err_eps_bad);
    fclose(f_iter_eps_good);
    fclose(f_iter_eps_bad);
    fclose(f_err_iter_good);
    fclose(f_err_iter_bad);

    free_matrix(A);
    free_matrix(A_good);
    free_matrix(A_bad);
    free_vector(b);
    free_vector(b_good);
    free_vector(b_bad);
    free_vector(x_exact);
    free_vector(x_sol);
    free_vector(error_vec);

    printf("Finished all experiments.\n");
    return 0;
}