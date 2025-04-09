#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // For memcpy
#include <stdbool.h> // For bool type

#define MAX_SIZE 10     
#define MAX_ITER 2000   
#define PERTURB_RUNS 20
#define SMALL_NUM 1e-15

void matrix_multiply(double A[MAX_SIZE][MAX_SIZE], double B[MAX_SIZE][MAX_SIZE], double C[MAX_SIZE][MAX_SIZE], int n);
void transpose_matrix(double A[MAX_SIZE][MAX_SIZE], double AT[MAX_SIZE][MAX_SIZE], int n);
int gram_schmidt(double A[MAX_SIZE][MAX_SIZE], double Q[MAX_SIZE][MAX_SIZE], int n);
int lu_decompose_pivot(double A[MAX_SIZE][MAX_SIZE], double LU[MAX_SIZE][MAX_SIZE], int P[MAX_SIZE], int n);
void lu_solve_pivot(double LU[MAX_SIZE][MAX_SIZE], int P[MAX_SIZE], double b[MAX_SIZE], double x[MAX_SIZE], int n);
void generate_random_matrix(double A[MAX_SIZE][MAX_SIZE], int n);

// --- Basic Vector/Matrix Utilities ---

// Print vector
void print_vector(double v[], int n, const char* name) {
    printf("%s: [", name);
    for (int i = 0; i < n; ++i) {
        printf("%8.4f%s", v[i], (i == n - 1) ? "" : ", ");
    }
    printf("]\n");
}

// Print matrix
void print_matrix(double A[MAX_SIZE][MAX_SIZE], int n, const char* name) {
    printf("%s (%dx%d):\n", name, n, n);
    for (int i = 0; i < n; ++i) {
        printf("  [");
        for (int j = 0; j < n; ++j) {
            printf("%8.4f%s", A[i][j], (j == n - 1) ? "" : ", ");
        }
        printf("]\n");
    }
}

// Calculate L2 norm of a vector
double vector_norm(double v[], int n) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_sq += v[i] * v[i];
    }
    return sqrt(sum_sq);
}

// Normalize a vector in place
void normalize_vector(double v[], int n) {
    double norm = vector_norm(v, n);
    if (norm < 1e-15) { // Avoid division by zero
        // Handle potentially zero vector (e.g., set first element to 1)
         // In practice, this indicates an issue, but let's prevent crashing.
        // A better approach might be to return an error.
        // printf("Warning: Attempting to normalize near-zero vector.\n");
        if (n > 0) v[0] = 1.0;
        for (int i = 1; i < n; ++i) v[i] = 0.0; // Zero out rest
        return;
    }
    for (int i = 0; i < n; ++i) {
        v[i] /= norm;
    }
}

// Copy vector src to dest
void copy_vector(double src[], double dest[], int n) {
    for (int i = 0; i < n; ++i) {
        dest[i] = src[i];
    }
}

// Matrix-vector multiplication: y = A * x
void matrix_vector_multiply(double A[MAX_SIZE][MAX_SIZE], double x[], double y[], int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            y[i] += A[i][j] * x[j];
        }
    }
}

// Vector subtraction: result = a - b
void vector_subtract(double a[], double b[], double result[], int n) {
    for (int i = 0; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
}

// Dot product: v1 . v2
double dot_product(double v1[], double v2[], int n) {
    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

// --- Linear System Solver (Gaussian Elimination with Partial Pivoting) ---
// Solves Ax = b, modifies A and b
// Returns 0 on success, 1 on failure (singular matrix)
int solve_linear_system(double A[MAX_SIZE][MAX_SIZE], double b[MAX_SIZE], double x[MAX_SIZE], int n) {
    double temp_A[MAX_SIZE][MAX_SIZE];
    double temp_b[MAX_SIZE];
    int pivot_row;
    double max_val, temp, multiplier;

    // Create copies to avoid modifying original A and b
    for (int i = 0; i < n; ++i) {
        temp_b[i] = b[i];
        for (int j = 0; j < n; ++j) {
            temp_A[i][j] = A[i][j];
        }
    }

    // Forward Elimination with Partial Pivoting
    for (int k = 0; k < n - 1; ++k) {
        pivot_row = k;
        max_val = fabs(temp_A[k][k]);
        for (int i = k + 1; i < n; ++i) {
            if (fabs(temp_A[i][k]) > max_val) {
                max_val = fabs(temp_A[i][k]);
                pivot_row = i;
            }
        }

        if (max_val < SMALL_NUM) { return 1; } // Singular

        if (pivot_row != k) {
            for (int j = k; j < n; ++j) { // Swap full rows in temp_A
                temp = temp_A[k][j];
                temp_A[k][j] = temp_A[pivot_row][j];
                temp_A[pivot_row][j] = temp;
            }
            temp = temp_b[k]; // Swap corresponding elements in temp_b
            temp_b[k] = temp_b[pivot_row];
            temp_b[pivot_row] = temp;
        }

        for (int i = k + 1; i < n; ++i) {
            if (fabs(temp_A[k][k]) < SMALL_NUM) return 1; // Avoid division by zero
            multiplier = temp_A[i][k] / temp_A[k][k];
            temp_A[i][k] = 0.0; // Store multiplier here if making LU in-place, otherwise just zero
            for (int j = k + 1; j < n; ++j) {
                temp_A[i][j] -= multiplier * temp_A[k][j];
            }
            temp_b[i] -= multiplier * temp_b[k];
        }
    }

    if (fabs(temp_A[n - 1][n - 1]) < SMALL_NUM) { return 1; } // Singular

    // Back Substitution
    for (int i = n - 1; i >= 0; --i) {
        x[i] = temp_b[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= temp_A[i][j] * x[j];
        }
        if (fabs(temp_A[i][i]) < SMALL_NUM) return 1; // Avoid division by zero
        x[i] /= temp_A[i][i];
    }
    return 0; // Success
}

// --- NEW: Matrix Multiplication ---
void matrix_multiply(double A[MAX_SIZE][MAX_SIZE], double B[MAX_SIZE][MAX_SIZE], double C[MAX_SIZE][MAX_SIZE], int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// --- NEW: Matrix Transpose ---
void transpose_matrix(double A[MAX_SIZE][MAX_SIZE], double AT[MAX_SIZE][MAX_SIZE], int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            AT[j][i] = A[i][j];
        }
    }
}

// --- NEW: Gram-Schmidt Orthogonalization ---
// Takes matrix A, outputs orthogonal matrix Q (columns are orthogonal vectors)
// Returns 0 on success, 1 if columns are linearly dependent
int gram_schmidt(double A[MAX_SIZE][MAX_SIZE], double Q[MAX_SIZE][MAX_SIZE], int n) {
    double v[MAX_SIZE];
    double proj_coeff;
    double temp_vec[MAX_SIZE];

    for (int j = 0; j < n; ++j) { // For each column j (vector a_j)
        // Start with v_j = a_j
        for (int i = 0; i < n; ++i) {
            v[i] = A[i][j];
        }

        // Subtract projections onto previous q_k vectors (k < j)
        for (int k = 0; k < j; ++k) {
            // Get q_k (k-th column of Q)
            for (int i = 0; i < n; ++i) temp_vec[i] = Q[i][k]; // Use temp_vec for q_k

            // Calculate projection coefficient: (a_j . q_k) / (q_k . q_k)
            // Since q_k will be normalized, (q_k . q_k) = 1
            proj_coeff = 0.0;
            for (int i = 0; i < n; ++i) proj_coeff += A[i][j] * temp_vec[i]; // a_j . q_k

            // Subtract projection: v = v - proj_coeff * q_k
            for (int i = 0; i < n; ++i) {
                v[i] -= proj_coeff * temp_vec[i];
            }
        }

        // Normalize v to get q_j
        double norm_v = vector_norm(v, n);
        if (norm_v < SMALL_NUM) {
            fprintf(stderr, "Error: Gram-Schmidt failed. Vectors are likely linearly dependent (column %d).\n", j);
            // Fill Q with something reasonable or just return error
            // For simplicity, let's make it identity up to this point and signal error
            for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c) Q[r][c] = (r == c) ? 1.0 : 0.0;
            return 1; // Indicate failure
        }
        for (int i = 0; i < n; ++i) {
            Q[i][j] = v[i] / norm_v;
        }
    }
    return 0; // Success
}


// --- NEW: LU Decomposition with Pivoting (Doolittle variant) ---
// Decomposes A into P*A = L*U.
// LU matrix is stored in place in LU argument (L has ones on diagonal, not stored).
// P is the permutation vector (P[i] = row 'i' was swapped with row P[i]).
// Returns 0 on success, 1 on singularity.
int lu_decompose_pivot(double A[MAX_SIZE][MAX_SIZE], double LU[MAX_SIZE][MAX_SIZE], int P[MAX_SIZE], int n) {
    // Initialize LU with A and P as identity permutation
    for (int i = 0; i < n; ++i) {
        P[i] = i;
        for (int j = 0; j < n; ++j) {
            LU[i][j] = A[i][j];
        }
    }

    for (int k = 0; k < n - 1; ++k) {
        // Find pivot row
        int pivot_row = k;
        double max_val = fabs(LU[k][k]);
        for (int i = k + 1; i < n; ++i) {
            if (fabs(LU[i][k]) > max_val) {
                max_val = fabs(LU[i][k]);
                pivot_row = i;
            }
        }

        if (max_val < SMALL_NUM) {
            fprintf(stderr, "Warning: LU decomposition failed, matrix singular or nearly singular (pivot step %d).\n", k);
            return 1; // Singular
        }

        // Swap rows if necessary in LU and P
        if (pivot_row != k) {
            int temp_p = P[k]; P[k] = P[pivot_row]; P[pivot_row] = temp_p;
            for (int j = 0; j < n; ++j) { // Swap entire rows in LU
                double temp_lu = LU[k][j];
                LU[k][j] = LU[pivot_row][j];
                LU[pivot_row][j] = temp_lu;
            }
        }

        // Elimination: Calculate L elements (below diagonal) and update U elements
        if (fabs(LU[k][k]) < SMALL_NUM) return 1; // Check again after swap? Should be redundant if max_val check passed.

        for (int i = k + 1; i < n; ++i) {
            LU[i][k] /= LU[k][k]; // Store multiplier (L element) in lower part
            for (int j = k + 1; j < n; ++j) {
                LU[i][j] -= LU[i][k] * LU[k][j]; // Update U element
            }
        }
    }
    // Check last diagonal element of U
    if (fabs(LU[n - 1][n - 1]) < SMALL_NUM) {
        fprintf(stderr, "Warning: LU decomposition failed, matrix singular or nearly singular (last pivot).\n");
        return 1;
    }

    return 0; // Success
}

// --- NEW: LU Solver using Decomposition ---
// Solves Ax = b where PA = LU. Needs LU matrix and permutation vector P.
// Solves L(Ux) = Pb using forward substitution for Ly = Pb, then back substitution for Ux = y.
void lu_solve_pivot(double LU[MAX_SIZE][MAX_SIZE], int P[MAX_SIZE], double b[MAX_SIZE], double x[MAX_SIZE], int n) {
    double y[MAX_SIZE]; // Intermediate vector

    // Forward substitution: Solve Ly = Pb
    for (int i = 0; i < n; ++i) {
        y[i] = b[P[i]]; // Apply permutation to b
        for (int j = 0; j < i; ++j) {
            y[i] -= LU[i][j] * y[j]; // Use L elements (below diagonal)
        }
        // Diagonal elements of L are 1 (not stored), so no division needed here.
    }

    // Back substitution: Solve Ux = y
    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= LU[i][j] * x[j]; // Use U elements (on and above diagonal)
        }
        if (fabs(LU[i][i]) < SMALL_NUM) { // Check for division by zero
            fprintf(stderr, "Error: Division by near-zero in lu_solve_pivot back substitution.\n");
            // Handle error: set x to NaN or zero? Or rely on decompose check?
            for (int k = 0; k < n; ++k) x[k] = NAN; // Not-a-Number
            return;
        }
        x[i] /= LU[i][i]; // Divide by diagonal element of U
    }
}


int inverse_iteration_with_shift(
    double A[MAX_SIZE][MAX_SIZE],
    int n,
    double initial_guess[],
    double initial_shift,
    double tolerance,
    int max_iterations,
    double* eigenvalue,
    double eigenvector[],
    int* iterations_count,
    double* error_history, // Can be NULL
    int error_history_size,
    bool use_variable_shift) // <<--- Changed to bool
{
    double B[MAX_SIZE][MAX_SIZE]; // Matrix B = A - shift*I
    double LU[MAX_SIZE][MAX_SIZE]; // Storage for LU decomposition if shift is constant
    int P[MAX_SIZE];               // Permutation vector for LU
    double z[MAX_SIZE];            // Solution of B*z = x or LU*z = Pb
    double x_current[MAX_SIZE];    // Current eigenvector estimate x_k
    double x_next[MAX_SIZE];       // Next eigenvector estimate x_k+1
    double residual[MAX_SIZE];     // Residual vector: A*x - lambda*x
    double current_shift = initial_shift;
    double lambda_est = initial_shift; // Start estimate near shift
    double residual_norm;
    int solve_status = 0;
    bool lu_decomposed = false;

    // Initialize eigenvector estimate
    copy_vector(initial_guess, x_current, n);
    normalize_vector(x_current, n);
    copy_vector(x_current, eigenvector, n); // Initial output guess

    *iterations_count = 0;
    *eigenvalue = lambda_est;

    // --- Pre-computation for Constant Shift ---
    if (!use_variable_shift) {
        // Construct B = A - shift * I
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                B[i][j] = A[i][j];
            }
            B[i][i] -= current_shift; // Use initial_shift (it's constant)
        }
        // Decompose B = LU (with pivoting)
        solve_status = lu_decompose_pivot(B, LU, P, n);
        if (solve_status != 0) {
            fprintf(stderr, "Error: LU decomposition failed for the constant shift %.4e. Cannot proceed.\n", current_shift);
            // Cannot recover if initial decomposition fails
            *eigenvalue = NAN;
            for (int i = 0; i < n; ++i) eigenvector[i] = NAN;
            return 2; // Signal solver failure
        }
        lu_decomposed = true;
        // printf("Constant shift: LU decomposition computed.\n");
    }
    // -----------------------------------------

    for (int k = 0; k < max_iterations; ++k) {
        *iterations_count = k + 1;

        // 1./2. Solve System
        if (use_variable_shift) {
            // --- Variable Shift: Recompute B and Solve ---
            // Construct B = A - current_shift * I
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    B[i][j] = A[i][j];
                }
                B[i][i] -= current_shift;
            }
            solve_status = solve_linear_system(B, x_current, z, n);
            if (solve_status != 0) {
                fprintf(stderr, "Warning: Linear system solve failed in iteration %d (variable shift=%.4e).\n", k + 1, current_shift);
                // Return current best estimate but signal issue
                // (lambda_est and x_current already hold previous step's result)
                *eigenvalue = lambda_est;
                copy_vector(x_current, eigenvector, n);
                return 2; // Signal solver failure
            }

        }
        else {
            // --- Constant Shift: Use Precomputed LU ---
            if (!lu_decomposed) { // Should not happen if initial check passed
                fprintf(stderr, "Fatal Error: LU not decomposed for constant shift case!\n");
                return 2;
            }
            // Solve LU z = P x_current using forward/back substitution
            lu_solve_pivot(LU, P, x_current, z, n);
            // Note: lu_solve_pivot doesn't easily return errors currently, assumes decomposition was valid. Add error check if needed.

        }

        // 3. Normalize z to get x_next
        copy_vector(z, x_next, n);
        normalize_vector(x_next, n); // Normalization handles potential scaling issues from solver

        // Check for potential NaN/Inf from solver/normalization
        bool valid_vector = true;
        for (int i = 0; i < n; ++i) {
            if (isnan(x_next[i]) || isinf(x_next[i])) {
                valid_vector = false;
                break;
            }
        }
        if (!valid_vector) {
            fprintf(stderr, "Error: NaN or Inf encountered in eigenvector estimate at iteration %d.\n", k + 1);
            *eigenvalue = NAN; // Signal error
            copy_vector(x_current, eigenvector, n); // Return previous best
            return 2; // Treat as solver failure
        }


        // 4. Estimate eigenvalue using Rayleigh Quotient
        matrix_vector_multiply(A, x_next, residual, n); // residual = A*x_next temporarily
        lambda_est = dot_product(x_next, residual, n); // R.Q. = x_next^T * A * x_next (since x_next is normalized)

        // Check for NaN/Inf in eigenvalue estimate
        if (isnan(lambda_est) || isinf(lambda_est)) {
            fprintf(stderr, "Error: NaN or Inf encountered in eigenvalue estimate at iteration %d.\n", k + 1);
            *eigenvalue = NAN;
            copy_vector(x_next, eigenvector, n); // Return current vector attempt
            return 1; // Treat as convergence failure / numerical issue
        }


        // 5. Calculate residual norm: ||A*x_next - lambda_est*x_next||
        for (int i = 0; i < n; ++i) {
            residual[i] -= lambda_est * x_next[i]; // residual = A*x_next - lambda*x_next
        }
        residual_norm = vector_norm(residual, n);

        // Store error if requested
        if (error_history != NULL && k < error_history_size) {
            error_history[k] = residual_norm;
        }

        // Update current vector for next iteration
        copy_vector(x_next, x_current, n);
        *eigenvalue = lambda_est; // Update eigenvalue estimate

        // 6. Check convergence
        if (residual_norm < tolerance) {
            copy_vector(x_current, eigenvector, n); // Store final vector
            return 0; // Success
        }

        // 7. Update shift if variable shifts are enabled
        if (use_variable_shift) {
            current_shift = lambda_est;
        }

        // Update eigenvector output on each iteration (best estimate so far)
        copy_vector(x_current, eigenvector, n);
    }

    // If loop finishes without converging
    fprintf(stderr, "Warning: Inverse iteration did not converge within %d iterations (shift=%.4e). Residual Norm = %e\n", max_iterations, initial_shift, residual_norm);
    copy_vector(x_current, eigenvector, n); // Store final vector anyway
    return 1; // Convergence failure
}

// --- NEW: Generate Random Matrix ---
void generate_random_matrix(double A[MAX_SIZE][MAX_SIZE], int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // Generate random double between -1.0 and 1.0
            A[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
}

// --- Helper to calculate eigenvector error ---
// Assumes v1 and v2 are normalized
double eigenvector_error(double v1[], double v2[], int n) {
    double dp = fabs(dot_product(v1, v2, n));
    // Error based on angle: sqrt(2*(1 - |v1.v2|)) or simply 1 - |v1.v2|
    // Or Euclidean distance: ||v1 - (+/-)v2||
    double err = 0;
    double diff[MAX_SIZE];
    // Try both v2 and -v2, take the minimum distance
    vector_subtract(v1, v2, diff, n);
    double err1 = vector_norm(diff, n);
    for (int i = 0; i < n; ++i) diff[i] = v1[i] + v2[i]; // v1 - (-v2)
    double err2 = vector_norm(diff, n);

    return fmin(err1, err2);

    // Alternative: return 1.0 - dp; // Error based on cosine similarity
}

int main() {
    srand(time(NULL));

    int n = MAX_SIZE; // Use the defined size (10)

    printf("Starting Eigenvalue Lab Computations (n=%d)\n", n);

    // --- Generate Test Matrices ---
    double D_good[MAX_SIZE][MAX_SIZE] = { 0 }; // Diagonal matrix for good sep.
    double D_bad[MAX_SIZE][MAX_SIZE] = { 0 };  // Diagonal matrix for bad sep.
    double R[MAX_SIZE][MAX_SIZE];            // Random matrix
    double Q[MAX_SIZE][MAX_SIZE];            // Orthogonal matrix from R
    double QT[MAX_SIZE][MAX_SIZE];           // Transpose of Q
    double Temp[MAX_SIZE][MAX_SIZE];         // Temporary for multiplication
    double A_good[MAX_SIZE][MAX_SIZE];       // Resulting good sep. matrix
    double A_bad[MAX_SIZE][MAX_SIZE];        // Resulting bad sep. matrix

    double eigvals_good[MAX_SIZE];
    double eigvals_bad[MAX_SIZE];
    // Eigenvectors will be the columns of Q

    printf("Generating %dx%d test matrices...\n", n, n);

    // 1. Define desired eigenvalues
    double target_val_good = 1.0; // Let's target the lowest eigenvalue
    double target_val_bad = 1.0;
    eigvals_good[0] = target_val_good;
    eigvals_bad[0] = target_val_bad;
    D_good[0][0] = target_val_good;
    D_bad[0][0] = target_val_bad;

    // Good separability: 1, 2, 3, ..., n
    for (int i = 1; i < n; ++i) {
        eigvals_good[i] = (double)(i + 1);
        D_good[i][i] = eigvals_good[i];
    }

    // Bad separability: 1, 1.01, 3, 4, ..., n
    eigvals_bad[1] = target_val_bad + 0.01; // Close eigenvalue
    D_bad[1][1] = eigvals_bad[1];
    for (int i = 2; i < n; ++i) {
        eigvals_bad[i] = (double)(i + 1);
        D_bad[i][i] = eigvals_bad[i];
    }

    // 2. Generate a random matrix R
    generate_random_matrix(R, n);

    // 3. Orthonormalize R using Gram-Schmidt to get Q
    if (gram_schmidt(R, Q, n) != 0) {
        fprintf(stderr, "FATAL: Failed to generate orthogonal matrix Q via Gram-Schmidt.\n");
        return 1;
    }
    // printf("Orthogonal matrix Q generated.\n");
    // print_matrix(Q, n, "Q");

    // 4. Calculate Q transpose
    transpose_matrix(Q, QT, n);

    // 5. Calculate A_good = Q * D_good * Q^T
    matrix_multiply(D_good, QT, Temp, n); // Temp = D_good * Q^T
    matrix_multiply(Q, Temp, A_good, n);   // A_good = Q * Temp
    printf("Matrix A_good (Good Separability) generated.\n");
    // print_matrix(A_good, n, "A_good");

    // 6. Calculate A_bad = Q * D_bad * Q^T
    matrix_multiply(D_bad, QT, Temp, n);   // Temp = D_bad * Q^T
    matrix_multiply(Q, Temp, A_bad, n);     // A_bad = Q * Temp
    printf("Matrix A_bad (Bad Separability) generated.\n");
    // print_matrix(A_bad, n, "A_bad");

    // --- General Parameters ---
    double initial_guess[MAX_SIZE];
    for (int i = 0; i < n; ++i) initial_guess[i] = (double)rand() / RAND_MAX; // Random initial guess
    normalize_vector(initial_guess, n); // Normalize initial guess

    double computed_eigenvalue;
    double computed_eigenvector[MAX_SIZE];
    int iterations;
    double eigenvalue_error_val;
    double eigenvector_error_val;
    double errors_history[MAX_ITER];

    // --- Target Eigenvalue/vector for error calculation (first one, eigenvalue = 1.0) ---
    double target_eigenvalue_good = eigvals_good[0];
    double target_eigenvector_good[MAX_SIZE]; // First column of Q
    for (int i = 0; i < n; ++i) target_eigenvector_good[i] = Q[i][0];

    double target_eigenvalue_bad = eigvals_bad[0];
    double target_eigenvector_bad[MAX_SIZE]; // First column of Q
    for (int i = 0; i < n; ++i) target_eigenvector_bad[i] = Q[i][0];

    // Shift close to the target eigenvalue (1.0)
    double shift = 0.9; // Good initial shift for target=1.0
    double default_tolerance = 1e-9; // Slightly tighter tolerance for larger matrices maybe


    // --- Investigation 1: Error vs Separability ---
    printf("\nInvestigation 1: Error vs Separability (Target EV ~ %.2f)\n", target_val_good);
    FILE* f_sep = fopen("separability_error.txt", "w");
    if (!f_sep) { perror("Error opening separability_error.txt"); return 1; }
    fprintf(f_sep, "SeparabilityType\tEigenvalueError\tEigenvectorError\tIterations\n");

    // Good separability (Constant Shift)
    inverse_iteration_with_shift(A_good, n, initial_guess, shift, default_tolerance, MAX_ITER,
        &computed_eigenvalue, computed_eigenvector, &iterations, NULL, 0, false); // Constant shift
    eigenvalue_error_val = fabs(computed_eigenvalue - target_eigenvalue_good);
    eigenvector_error_val = eigenvector_error(computed_eigenvector, target_eigenvector_good, n);
    fprintf(f_sep, "Good\t%.10e\t%.10e\t%d\n", eigenvalue_error_val, eigenvector_error_val, iterations);
    printf("  Good Separability: EV_err=%.2e, Vec_err=%.2e, Iters=%d\n", eigenvalue_error_val, eigenvector_error_val, iterations);

    // Bad separability (Constant Shift)
    // Use a different initial guess? Maybe not necessary if shift is good.
    // for (int i = 0; i < n; ++i) initial_guess[i] = (double)rand() / RAND_MAX; normalize_vector(initial_guess, n);
    inverse_iteration_with_shift(A_bad, n, initial_guess, shift, default_tolerance, MAX_ITER,
        &computed_eigenvalue, computed_eigenvector, &iterations, NULL, 0, false); // Constant shift
    eigenvalue_error_val = fabs(computed_eigenvalue - target_eigenvalue_bad);
    eigenvector_error_val = eigenvector_error(computed_eigenvector, target_eigenvector_bad, n);
    fprintf(f_sep, "Bad\t%.10e\t%.10e\t%d\n", eigenvalue_error_val, eigenvector_error_val, iterations);
    printf("  Bad Separability:  EV_err=%.2e, Vec_err=%.2e, Iters=%d\n", eigenvalue_error_val, eigenvector_error_val, iterations);
    fclose(f_sep);


    // --- Investigation 2: Error vs Tolerance ---
    printf("\nInvestigation 2: Error vs Tolerance (Good Separability Matrix, Constant Shift)\n");
    FILE* f_tol = fopen("tolerance_error.txt", "w");
    if (!f_tol) { perror("Error opening tolerance_error.txt"); return 1; }
    fprintf(f_tol, "Tolerance\tEigenvalueError\tEigenvectorError\tIterations\n");

    for (double tol = 1e-3; tol >= 1e-13; tol *= 1e-1) { // Adjusted tolerance range
        inverse_iteration_with_shift(A_good, n, initial_guess, shift, tol, MAX_ITER,
            &computed_eigenvalue, computed_eigenvector, &iterations, NULL, 0, false);
        eigenvalue_error_val = fabs(computed_eigenvalue - target_eigenvalue_good);
        eigenvector_error_val = eigenvector_error(computed_eigenvector, target_eigenvector_good, n);
        fprintf(f_tol, "%.1e\t%.10e\t%.10e\t%d\n", tol, eigenvalue_error_val, eigenvector_error_val, iterations);
        printf("  Tolerance=%.1e: EV_err=%.2e, Vec_err=%.2e, Iters=%d\n", tol, eigenvalue_error_val, eigenvector_error_val, iterations);
    }
    fclose(f_tol);


    // --- Investigation 3: Iterations vs Tolerance (Good/Bad Separability, Constant Shift) ---
    printf("\nInvestigation 3: Iterations vs Tolerance (Constant Shift)\n");
    FILE* f_iter_tol = fopen("iterations_tolerance.txt", "w");
    if (!f_iter_tol) { perror("Error opening iterations_tolerance.txt"); return 1; }
    fprintf(f_iter_tol, "SeparabilityType\tTolerance\tIterations\n");

    for (double tol = 1e-3; tol >= 1e-13; tol *= 1e-1) {
        // Good
        inverse_iteration_with_shift(A_good, n, initial_guess, shift, tol, MAX_ITER,
            &computed_eigenvalue, computed_eigenvector, &iterations, NULL, 0, false);
        fprintf(f_iter_tol, "Good\t%.1e\t%d\n", tol, iterations);
        printf("  Good, Tol=%.1e: Iters=%d\n", tol, iterations);
        // Bad
        inverse_iteration_with_shift(A_bad, n, initial_guess, shift, tol, MAX_ITER,
            &computed_eigenvalue, computed_eigenvector, &iterations, NULL, 0, false);
        fprintf(f_iter_tol, "Bad\t%.1e\t%d\n", tol, iterations);
        printf("  Bad,  Tol=%.1e: Iters=%d\n", tol, iterations);
    }
    fclose(f_iter_tol);


    // --- Investigation 4: Error vs Iteration Number ---
    printf("\nInvestigation 4: Error History (Good Separability Matrix, Constant Shift)\n");
    FILE* f_err_hist = fopen("error_vs_iteration.txt", "w");
    if (!f_err_hist) { perror("Error opening error_vs_iteration.txt"); return 1; }
    fprintf(f_err_hist, "Iteration\tResidualNorm\n");

    inverse_iteration_with_shift(A_good, n, initial_guess, shift, default_tolerance, MAX_ITER,
        &computed_eigenvalue, computed_eigenvector, &iterations,
        errors_history, MAX_ITER, false); // Get history, constant shift

    printf("  Saving error history for %d iterations...\n", iterations);
    for (int i = 0; i < iterations; ++i) {
        fprintf(f_err_hist, "%d\t%.10e\n", i + 1, errors_history[i]);
    }
    fclose(f_err_hist);


    // --- Investigation 5: Error vs Matrix Perturbation ---
    printf("\nInvestigation 5: Error vs Perturbation (Good Separability Matrix, Constant Shift)\n");
    FILE* f_pert = fopen("perturbation_error.txt", "w");
    if (!f_pert) { perror("Error opening perturbation_error.txt"); return 1; }
    fprintf(f_pert, "PerturbationLevel\tRun\tEigenvalueError\tEigenvectorError\tIterations\n");

    double A_perturbed[MAX_SIZE][MAX_SIZE];
    double perturbation_levels[] = { 0.01, 0.02, 0.03, 0.04, 0.05 }; // 1% to 5%
    int num_levels = sizeof(perturbation_levels) / sizeof(perturbation_levels[0]);
    int perturb_row_index = 0; // Perturb the first row

    // Find max absolute value in the *original good* matrix for scaling
    double max_abs_val = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fabs(A_good[i][j]) > max_abs_val) {
                max_abs_val = fabs(A_good[i][j]);
            }
        }
    }
    if (max_abs_val < SMALL_NUM) max_abs_val = 1.0; // Avoid division by zero

    printf("  Perturbing row %d of the 'Good' matrix (MaxAbsVal=%.2e)...\n", perturb_row_index, max_abs_val);
    for (int level_idx = 0; level_idx < num_levels; ++level_idx) {
        double p_level = perturbation_levels[level_idx];
        printf("  Perturbation Level: %.2f%%\n", p_level * 100.0);

        for (int run = 0; run < PERTURB_RUNS; ++run) {
            // Copy original good matrix
            memcpy(A_perturbed, A_good, sizeof(A_good));

            // Add perturbation to the specified row
            for (int j = 0; j < n; ++j) {
                double noise = ((double)rand() / RAND_MAX - 0.5) * 2.0; // Random number in [-1, 1]
                A_perturbed[perturb_row_index][j] += p_level * max_abs_val * noise;
            }

            // Run inverse iteration (constant shift) on perturbed matrix
            inverse_iteration_with_shift(A_perturbed, n, initial_guess, shift, default_tolerance, MAX_ITER,
                &computed_eigenvalue, computed_eigenvector, &iterations, NULL, 0, false);

            // Calculate error relative to the *original* eigenvalue/vector
            eigenvalue_error_val = fabs(computed_eigenvalue - target_eigenvalue_good);
            eigenvector_error_val = eigenvector_error(computed_eigenvector, target_eigenvector_good, n);

            fprintf(f_pert, "%.3f\t%d\t%.10e\t%.10e\t%d\n",
                p_level, run + 1, eigenvalue_error_val, eigenvector_error_val, iterations);
        }
    }
    fclose(f_pert);


    // --- Investigation 6: Constant vs Variable Shifts ---
    printf("\nInvestigation 6: Constant vs Variable Shifts\n");
    FILE* f_shift_comp = fopen("shift_comparison.txt", "w"); // Renamed file pointer
    if (!f_shift_comp) { perror("Error opening shift_comparison.txt"); return 1; }
    fprintf(f_shift_comp, "SeparabilityType\tShiftType\tIterations\tEigenvalueError\tEigenvectorError\n");

    // Good Separability
    printf("  Good Separability Matrix:\n");
    // Constant Shift (already computed in Inv 1, recompute for consistency?)
    inverse_iteration_with_shift(A_good, n, initial_guess, shift, default_tolerance, MAX_ITER,
        &computed_eigenvalue, computed_eigenvector, &iterations, NULL, 0, false); // variable_shift=false
    eigenvalue_error_val = fabs(computed_eigenvalue - target_eigenvalue_good);
    eigenvector_error_val = eigenvector_error(computed_eigenvector, target_eigenvector_good, n);
    fprintf(f_shift_comp, "Good\tConstant\t%d\t%.10e\t%.10e\n", iterations, eigenvalue_error_val, eigenvector_error_val);
    printf("    Constant Shift: Iters=%d, EV_err=%.2e, Vec_err=%.2e\n", iterations, eigenvalue_error_val, eigenvector_error_val);

    // Variable Shift
    inverse_iteration_with_shift(A_good, n, initial_guess, shift, default_tolerance, MAX_ITER,
        &computed_eigenvalue, computed_eigenvector, &iterations, NULL, 0, true); // variable_shift=true
    eigenvalue_error_val = fabs(computed_eigenvalue - target_eigenvalue_good);
    eigenvector_error_val = eigenvector_error(computed_eigenvector, target_eigenvector_good, n);
    fprintf(f_shift_comp, "Good\tVariable\t%d\t%.10e\t%.10e\n", iterations, eigenvalue_error_val, eigenvector_error_val);
    printf("    Variable Shift: Iters=%d, EV_err=%.2e, Vec_err=%.2e\n", iterations, eigenvalue_error_val, eigenvector_error_val);

    // Bad Separability
    printf("  Bad Separability Matrix:\n");
    // Constant Shift (already computed in Inv 1)
    inverse_iteration_with_shift(A_bad, n, initial_guess, shift, default_tolerance, MAX_ITER,
        &computed_eigenvalue, computed_eigenvector, &iterations, NULL, 0, false); // variable_shift=false
    eigenvalue_error_val = fabs(computed_eigenvalue - target_eigenvalue_bad);
    eigenvector_error_val = eigenvector_error(computed_eigenvector, target_eigenvector_bad, n);
    fprintf(f_shift_comp, "Bad\tConstant\t%d\t%.10e\t%.10e\n", iterations, eigenvalue_error_val, eigenvector_error_val);
    printf("    Constant Shift: Iters=%d, EV_err=%.2e, Vec_err=%.2e\n", iterations, eigenvalue_error_val, eigenvector_error_val);

    // Variable Shift
    inverse_iteration_with_shift(A_bad, n, initial_guess, shift, default_tolerance, MAX_ITER,
        &computed_eigenvalue, computed_eigenvector, &iterations, NULL, 0, true); // variable_shift=true
    // Check which nearby eigenvalue it converged to (1.0 or 1.01)
    double target_ev_bad_0 = eigvals_bad[0]; // Should be 1.0
    double target_ev_bad_1 = eigvals_bad[1]; // Should be 1.01
    double target_vec_bad_0[MAX_SIZE]; for (int i = 0; i < n; ++i) target_vec_bad_0[i] = Q[i][0];
    double target_vec_bad_1[MAX_SIZE]; for (int i = 0; i < n; ++i) target_vec_bad_1[i] = Q[i][1];

    double err_to_ev0 = fabs(computed_eigenvalue - target_ev_bad_0);
    double err_to_ev1 = fabs(computed_eigenvalue - target_ev_bad_1);

    if (err_to_ev0 < err_to_ev1) {
        eigenvalue_error_val = err_to_ev0;
        eigenvector_error_val = eigenvector_error(computed_eigenvector, target_vec_bad_0, n);
        printf("    (Variable shift converged to EV=%.2f)\n", target_ev_bad_0);
    }
    else {
        eigenvalue_error_val = err_to_ev1;
        eigenvector_error_val = eigenvector_error(computed_eigenvector, target_vec_bad_1, n);
        printf("    (Variable shift likely converged to EV=%.2f)\n", target_ev_bad_1);
    }
    fprintf(f_shift_comp, "Bad\tVariable\t%d\t%.10e\t%.10e\n", iterations, eigenvalue_error_val, eigenvector_error_val);
    printf("    Variable Shift: Iters=%d, EV_err=%.2e, Vec_err=%.2e\n", iterations, eigenvalue_error_val, eigenvector_error_val);

    fclose(f_shift_comp);


    printf("\nComputations finished. Data saved to .txt files.\n");
    return 0;
}