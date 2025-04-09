#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h> // For DBL_MAX

#define MAX_ITER 100
#define HIGH_PRECISION_TOL 1e-16
#define PERTURB_SAMPLES 20
#define DATA_POINTS_PLOT 200 // Number of points for function plots
#define NUM_FUNCS 3

// --- Function Definitions ---

// Equation 1: 0.1*x^2 - x*ln(x) = 0
double f1(double x) {
    if (x <= 0) { // Avoid log(0) or log(-ve)
        return DBL_MAX; // Indicate invalid input
    }
    return 0.1 * x * x - x * log(x);
}

// Equation 2: x^4 - 3*x^2 + 75*x - 9999 = 0
double f2(double x) {
    return pow(x, 4) - 3.0 * pow(x, 2) + 75.0 * x - 9999.0;
}

// Equation 3: f1 with discontinuity at x = 1.5
double f3(double x) {
    if (x <= 0) return DBL_MAX;
    if (fabs(x - 1.5) < 1e-10) return DBL_MAX;
    return 0.1 * x * x - x * log(x);
}

// Function pointer type
typedef double (*FuncPtr)(double);

// --- Perturbation Globals ---
// Global variables to control perturbation (simpler for this specific case)
int g_perturb_active = 0; // 0: inactive, 1: active
double g_perturb_percentage = 0.0;
double g_perturb_coeff_original_value = 0.0;

// Equation 2 with perturbation on the constant term (-9999)
double f2_perturbed(double x) {
    double coeff = -9999.0;
    if (g_perturb_active) {
        // Generate random perturbation within [-percentage, +percentage]
        double random_factor = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Range [-1, 1]
        double perturbation = g_perturb_percentage * g_perturb_coeff_original_value * random_factor;
        coeff += perturbation;
    }
    return pow(x, 4) - 3.0 * pow(x, 2) + 75.0 * x + coeff;
}


// --- Root Finding Methods ---

// Bisection Method
// Returns 0 on success, 1 if max iterations reached, 2 if f(a)*f(b)>0, 3 on other errors
int bisection(FuncPtr f, double a, double b, double tol, int max_iter, double* root, int* iterations, double errors_iter[], double true_root) {
    double fa = f(a);
    double fb = f(b);

    if (fa == DBL_MAX || fb == DBL_MAX) return 3;
    if (fa * fb >= 0) {
        if (fabs(fa) < tol) { *root = a; *iterations = 0; return 0; }
        if (fabs(fb) < tol) { *root = b; *iterations = 0; return 0; }
        return 2;
    }

    double c, fc;
    for (int i = 0; i < max_iter; ++i) {
        c = (a + b) / 2;
        fc = f(c);

        if (errors_iter) errors_iter[i] = fabs(c - true_root);

        if (fc == DBL_MAX) return 3;
        if (fabs(fc) < tol || (b - a) / 2 < tol) {
            *root = c;
            *iterations = i + 1;
            return 0;
        }

        if (fa * fc < 0) { b = c; fb = fc; }
        else { a = c; fa = fc; }
    }
    *root = c;
    *iterations = max_iter;
    return 1;
}

// Returns 0 on success, 1 if max iterations reached, 2 if division by zero, 3 on other errors
int Hord(FuncPtr f, double a, double b, double tol, int max_iter, double* root, int* iterations, double errors_iter[], double true_root) {
    double fa = f(a);
    double fb = f(b);

    if (fa == DBL_MAX || fb == DBL_MAX) return 3;
    if (fa * fb >= 0) return 2;

    // Determine fixed point (where sign matches second derivative)
    double x_fixed, f_fixed;
    double dx = 1e-6;
    double ddf_a = (f(a + dx) - 2 * f(a) + f(a - dx)) / (dx * dx);
    double ddf_b = (f(b + dx) - 2 * f(b) + f(b - dx)) / (dx * dx);

    if (fa * ddf_a > 0) { x_fixed = a; f_fixed = fa; }
    else { x_fixed = b; f_fixed = fb; }

    double x = (x_fixed == a) ? b : a;
    double fx = (x_fixed == a) ? fb : fa;

    for (int i = 0; i < max_iter; ++i) {
        double x_new = x - fx * (x - x_fixed) / (fx - f_fixed);
        double fx_new = f(x_new);

        if (errors_iter) errors_iter[i] = fabs(x_new - true_root);

        if (fx_new == DBL_MAX) return 3;
        if (fabs(x_new - x) < tol || fabs(fx_new) < tol) {
            *root = x_new;
            *iterations = i + 1;
            return 0;
        }

        x = x_new;
        fx = fx_new;
    }
    *root = x;
    *iterations = max_iter;
    return 1;
}

// --- Helper Functions ---

// Find intervals containing roots by simple step search
void find_intervals(FuncPtr f, double start, double end, double step, FILE* out_file) {
    fprintf(out_file, "Searching for intervals containing roots between %.2f and %.2f with step %.4f\n", start, end, step);
    double x1 = start;
    double y1 = f(x1);
    int count = 0;
    while (x1 < end) {
        double x2 = x1 + step;
        if (x2 > end) x2 = end;
        double y2 = f(x2);

        // Skip if function is discontinuous/invalid in the interval
        if (y1 == DBL_MAX || y2 == DBL_MAX) {
            x1 = x2;
            y1 = f(x1); // Re-evaluate y1 for the next interval start
            continue;
        }


        if (y1 * y2 < 0) {
            fprintf(out_file, "Potential root found in interval [%.4f, %.4f]\n", x1, x2);
            count++;
        }
        else if (fabs(y1) < 1e-9) { // Check if x1 itself is a root
            fprintf(out_file, "Potential root found at or very near x = %.4f\n", x1);
            count++;
        }
        else if (x2 == end && fabs(y2) < 1e-9) { // Check end point
            fprintf(out_file, "Potential root found at or very near x = %.4f\n", x2);
            count++;
        }


        x1 = x2;
        y1 = y2;
        if (x1 == end) break; // Avoid infinite loop if step doesn't perfectly reach end
    }
    if (count == 0) {
        fprintf(out_file, "No sign changes detected in the given range with this step size.\n");
    }
    fprintf(out_file, "----------------------------------------\n");
}


// Generate data for plotting a function
void generate_function_data(FuncPtr f, double start, double end, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file for function data");
        return;
    }
    fprintf(file, "x\ty\n"); // Header
    double step = (end - start) / (DATA_POINTS_PLOT - 1);
    for (int i = 0; i < DATA_POINTS_PLOT; ++i) {
        double x = start + i * step;
        double y = f(x);
        if (y != DBL_MAX) { // Only write valid points
            fprintf(file, "%.16f\t%.16f\n", x, y);
        }
        else {
            fprintf(file, "%.16f\tnan\n", x); // Indicate gap for plotting
        }
    }
    fclose(file);
}

// --- Main Analysis ---

int main() {
    srand(time(NULL)); // Seed random number generator for perturbations

    FILE* results_file = fopen("results.txt", "w");
    if (!results_file) {
        perror("Error opening results file");
        return 1;
    }

    fprintf(results_file, "Nonlinear Equation Solver Analysis\n");
    fprintf(results_file, "==================================\n\n");

    // --- Define functions and intervals/guesses ---
    FuncPtr functions[] = { f1, f2, f3 };
    const char* func_names[] = { "f1(x)=0.1x^2-xln(x)", "f2(x)=x^4-3x^2+75x-9999", "f3(x)=f1 with discontinuity" };
    double intervals[][2] = {
        {1.0, 2.0},    // f1
        {9.0, 11.0},   // f2
        {1.0, 1.4999}  // f3: root before discontinuity at 1.5
    };


    // --- Task 1: Equations defined (f1, f2, f3) ---
    fprintf(results_file, "Task 1: Equations Defined\n");
    fprintf(results_file, "- %s\n", func_names[0]);
    fprintf(results_file, "- %s\n", func_names[1]);
    fprintf(results_file, "- %s at x=1.5\n", func_names[2]);
    fprintf(results_file, "----------------------------------------\n\n");

    // --- Task 2: Root Isolation ---
    fprintf(results_file, "Task 2: Root Isolation\n\n");

    // Generate data for plotting functions
    fprintf(results_file, "Generating data for function plots...\n");
    generate_function_data(f1, 0.1, 3.0, "f1_data.txt");
    generate_function_data(f2, -15.0, 15.0, "f2_data.txt"); // Wider range for f2
    generate_function_data(f3, 0.1, 3.0, "f3_data.txt");
    fprintf(results_file, "- Data saved to f1_data.txt, f2_data.txt, f3_data.txt\n");
    fprintf(results_file, "- Plot these files using Python to visualize roots and discontinuity.\n\n");


    // Graphical illustration of merging roots with large step (for f2)
    fprintf(results_file, "Illustrating root separation sensitivity to step size (f2):\n");
    find_intervals(f2, 8.0, 12.0, 1.0, results_file); // Coarse step - might merge roots if close
    find_intervals(f2, 8.0, 12.0, 0.1, results_file); // Finer step - should separate
    // Add search for negative roots for f2
    find_intervals(f2, -12.0, -8.0, 0.1, results_file);

    // Analytical bounds (simple example for positive roots of f2: x^4-3x^2+75x-9999=0)
    // Using a simple method (e.g., based on coefficients)
    // For P(x) = a_n*x^n + ... - a_k*x^k + ... + a_0 = 0 (a_n > 0)
    // Upper bound B = 1 + (A / a_n)^(1/k) where A is max absolute value of negative coeffs, k is index of first negative coeff
    double coeffs_f2[] = { -9999.0, 75.0, -3.0, 0.0, 1.0 }; // a0, a1, a2, a3, a4
    int degree_f2 = 4;
    double max_neg_abs = 0;
    int first_neg_k = -1;
    for (int i = 0; i < degree_f2; ++i) { // Check coeffs a0 to a_{n-1}
        if (coeffs_f2[i] < 0) {
            if (first_neg_k == -1) first_neg_k = degree_f2 - i; // k = n - index
            if (fabs(coeffs_f2[i]) > max_neg_abs) {
                max_neg_abs = fabs(coeffs_f2[i]);
            }
        }
    }
    double upper_bound_pos = DBL_MAX;
    if (first_neg_k != -1 && coeffs_f2[degree_f2] > 0) {
        upper_bound_pos = 1.0 + pow(max_neg_abs / coeffs_f2[degree_f2], 1.0 / (double)first_neg_k);
        fprintf(results_file, "Analytical Upper Bound for Positive Roots of f2 (Method 1): %.4f\n", upper_bound_pos);
    }
    else {
        fprintf(results_file, "Analytical Upper Bound for Positive Roots of f2 (Method 1): No negative coefficients found (or leading coeff not positive), method not directly applicable or no positive roots guaranteed by this method.\n");
    }
    // Another simple bound: R = 1 + max(|a_i / a_n|) for i = 0 to n-1
    double max_ratio = 0;
    for (int i = 0; i < degree_f2; ++i) {
        if (fabs(coeffs_f2[i] / coeffs_f2[degree_f2]) > max_ratio) {
            max_ratio = fabs(coeffs_f2[i] / coeffs_f2[degree_f2]);
        }
    }
    double upper_bound_pos_neg = 1.0 + max_ratio;
    fprintf(results_file, "Analytical Upper Bound for Absolute Value of Roots of f2 (Cauchy): %.4f\n", upper_bound_pos_neg);
    fprintf(results_file, "=> Roots (if any) lie within [%.4f, %.4f]\n", -upper_bound_pos_neg, upper_bound_pos_neg);


    fprintf(results_file, "Selected intervals/guesses for analysis based on plots/search:\n");
    fprintf(results_file, "f1: [%.2f, %.2f]\n", intervals[0][0], intervals[0][1]);
    fprintf(results_file, "f2: [%.2f, %.2f] (for positive root)\n", intervals[1][0], intervals[1][1]);
    fprintf(results_file, "f3: [%.2f, %.2f] (before discontinuity)\n", intervals[2][0], intervals[2][1]);

    fprintf(results_file, "----------------------------------------\n\n");


    // --- Find high-precision roots first (to use as 'true' roots for error calculation) ---
    double true_roots[NUM_FUNCS];
    fprintf(results_file, "Calculating high-precision reference roots...\n");
    for (int i = 0; i < NUM_FUNCS; ++i) {
        int iter_tmp;
        // Use Hord method often converges faster for smooth functions
        int status = Hord(functions[i], intervals[i][0], intervals[i][1], HIGH_PRECISION_TOL, MAX_ITER, &true_roots[i], &iter_tmp, NULL, 0);
        if (status != 0 && i != 2) { // Allow f3 to potentially fail if interval choice was bad
            // If Hord fails, try bisection
            status = bisection(functions[i], intervals[i][0], intervals[i][1], HIGH_PRECISION_TOL, MAX_ITER, &true_roots[i], &iter_tmp, NULL, 0);
        }

        if (status == 0) {
            fprintf(results_file, "True root estimate for %s: %.16f (found in %d iterations)\n", func_names[i], true_roots[i], iter_tmp);
        }
        else {
            true_roots[i] = NAN; // Mark as not found
            fprintf(results_file, "Warning: Could not find high-precision root for %s (Status: %d). Error analysis might be inaccurate.\n", func_names[i], status);
            // Try to find *some* root for f3 even if high precision failed near discontinuity
            if (i == 2) {
                bisection(functions[i], 1.0, 1.4, 1e-8, MAX_ITER, &true_roots[i], &iter_tmp, NULL, 0);
                fprintf(results_file, "Attempted lower precision root for %s: %.8f\n", func_names[i], true_roots[i]);
            }

        }
    }
    fprintf(results_file, "----------------------------------------\n\n");


    // --- Task 3: Error vs. Iteration ---
    fprintf(results_file, "Task 3: Error vs. Iteration Analysis\n");
    double errors_iter_bisec[MAX_ITER];
    double errors_iter_Hord[MAX_ITER];
    int iterations_bisec, iterations_Hord;
    double root_bisec, root_Hord;

    for (int i = 0; i < NUM_FUNCS; ++i) {
        if (isnan(true_roots[i])) {
            fprintf(results_file, "Skipping Error vs Iteration for %s as true root not found.\n", func_names[i]);
            continue;
        }

        fprintf(results_file, "Running for %s...\n", func_names[i]);
        char filename_bisec[64], filename_Hord[64];
        sprintf(filename_bisec, "err_vs_iter_%s_bisec.txt", func_names[i]);
        sprintf(filename_Hord, "err_vs_iter_%s_Hord.txt", func_names[i]);

        FILE* fb = fopen(filename_bisec, "w");
        FILE* fs = fopen(filename_Hord, "w");
        if (!fb || !fs) {
            perror("Error opening error vs iteration file");
            continue;
        }
        fprintf(fb, "Iteration\tAbsoluteError\n");
        fprintf(fs, "Iteration\tAbsoluteError\n");

        // Run Bisection
        int status_b = bisection(functions[i], intervals[i][0], intervals[i][1], HIGH_PRECISION_TOL, MAX_ITER, &root_bisec, &iterations_bisec, errors_iter_bisec, true_roots[i]);
        if (status_b == 0 || status_b == 1) { // Success or max iter reached
            for (int k = 0; k < iterations_bisec; ++k) {
                fprintf(fb, "%d\t%.16e\n", k + 1, errors_iter_bisec[k]);
            }
            fprintf(results_file, "  Bisection for %s finished in %d iterations (Status %d). Data saved to %s\n", func_names[i], iterations_bisec, status_b, filename_bisec);
        }
        else {
            fprintf(results_file, "  Bisection failed for %s (Status %d).\n", func_names[i], status_b);
        }


        // Run Hord
        int status_s = Hord(functions[i], intervals[i][0], intervals[i][1], HIGH_PRECISION_TOL, MAX_ITER, &root_Hord, &iterations_Hord, errors_iter_Hord, true_roots[i]);
        if (status_s == 0 || status_s == 1) { // Success or max iter reached
            for (int k = 0; k < iterations_Hord; ++k) {
                fprintf(fs, "%d\t%.16e\n", k + 1, errors_iter_Hord[k]);
            }
            fprintf(results_file, "  Hord for %s finished in %d iterations (Status %d). Data saved to %s\n", func_names[i], iterations_Hord, status_s, filename_Hord);
        }
        else {
            fprintf(results_file, "  Hord failed for %s (Status %d).\n", func_names[i], status_s);
        }

        fclose(fb);
        fclose(fs);
    }
    fprintf(results_file, "----------------------------------------\n\n");


    // --- Task 4: Achieved Error vs. Requested Tolerance ---
    fprintf(results_file, "Task 4: Achieved Error vs. Requested Tolerance Analysis\n");
    double tolerances[] = { 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15 };
    int num_tols = sizeof(tolerances) / sizeof(tolerances[0]);

    for (int i = 0; i < NUM_FUNCS; ++i) {
        if (isnan(true_roots[i])) {
            fprintf(results_file, "Skipping Error vs Tolerance for %s as true root not found.\n", func_names[i]);
            continue;
        }

        fprintf(results_file, "Running for %s...\n", func_names[i]);
        char filename_bisec[50], filename_Hord[50];
        sprintf(filename_bisec, "err_vs_tol_%s_bisec.txt", func_names[i]);
        sprintf(filename_Hord, "err_vs_tol_%s_Hord.txt", func_names[i]);

        FILE* fb = fopen(filename_bisec, "w");
        FILE* fs = fopen(filename_Hord, "w");
        if (!fb || !fs) {
            perror("Error opening error vs tolerance file");
            continue;
        }
        fprintf(fb, "RequestedTolerance\tAchievedError\n");
        fprintf(fs, "RequestedTolerance\tAchievedError\n");

        for (int j = 0; j < num_tols; ++j) {
            double tol = tolerances[j];
            int iter_b, iter_s;
            double root_b, root_s;
            double achieved_err_b = NAN, achieved_err_s = NAN;

            int status_b = bisection(functions[i], intervals[i][0], intervals[i][1], tol, MAX_ITER, &root_b, &iter_b, NULL, true_roots[i]);
            if (status_b == 0) {
                achieved_err_b = fabs(root_b - true_roots[i]);
                fprintf(fb, "%.16e\t%.16e\n", tol, achieved_err_b);
            }
            else {
                fprintf(fb, "%.16e\t%.16e\n", tol, NAN); // Indicate failure for this tolerance
                fprintf(results_file, "  Warning: Bisection failed for %s with tol=%.1e (Status %d)\n", func_names[i], tol, status_b);
            }


            int status_s = Hord(functions[i], intervals[i][0], intervals[i][1], tol, MAX_ITER, &root_s, &iter_s, NULL, true_roots[i]);
            if (status_s == 0) {
                achieved_err_s = fabs(root_s - true_roots[i]);
                fprintf(fs, "%.16e\t%.16e\n", tol, achieved_err_s);
            }
            else {
                fprintf(fs, "%.16e\t%.16e\n", tol, NAN); // Indicate failure
                fprintf(results_file, "  Warning: Hord failed for %s with tol=%.1e (Status %d)\n", func_names[i], tol, status_s);
            }
        }
        fprintf(results_file, "  Data saved to %s and %s\n", filename_bisec, filename_Hord);
        fclose(fb);
        fclose(fs);
    }
    fprintf(results_file, "----------------------------------------\n\n");


    // --- Task 5: Relative Error vs. Input Perturbation ---
    fprintf(results_file, "Task 5: Relative Error vs. Input Perturbation (for f2, Hord Method)\n");
    int func_idx_perturb = 1; // Index of f2 in the functions array
    double root_unperturbed = true_roots[func_idx_perturb];

    if (isnan(root_unperturbed)) {
        fprintf(results_file, "Skipping Perturbation analysis for f2 as reference root not found.\n");
    }
    else {
        fprintf(results_file, "Reference (unperturbed) root for f2: %.16f\n", root_unperturbed);
        char filename_perturb[50];
        sprintf(filename_perturb, "perturb_%s_Hord.txt", func_names[func_idx_perturb]);
        FILE* fp = fopen(filename_perturb, "w");
        if (!fp) {
            perror("Error opening perturbation data file");
        }
        else {
            fprintf(fp, "PerturbationPercent\tRelativeError\n");
            g_perturb_coeff_original_value = -9999.0; // Constant term of f2

            for (int p = 1; p <= 5; ++p) { // Perturbation 1% to 5%
                double percent = (double)p / 100.0;
                g_perturb_percentage = percent; // Set global percentage
                fprintf(results_file, "  Running with perturbation: %d%%\n", p);

                for (int k = 0; k < PERTURB_SAMPLES; ++k) {
                    double root_perturbed;
                    int iter_p;
                    double err_rel = NAN;

                    // Activate perturbation for this run
                    g_perturb_active = 1;
                    // Use f2_perturbed which uses the global flags
                    int status_p = Hord(f2_perturbed, intervals[func_idx_perturb][0], intervals[func_idx_perturb][1], 1e-10, MAX_ITER, &root_perturbed, &iter_p, NULL, 0); // Don't need error array here
                    g_perturb_active = 0; // Deactivate for next sample/percentage


                    if (status_p == 0 && fabs(root_unperturbed) > DBL_EPSILON) { // Check status and avoid division by zero in relative error
                        err_rel = fabs(root_perturbed - root_unperturbed) / fabs(root_unperturbed);
                        fprintf(fp, "%.2f\t%.16e\n", (double)p, err_rel);
                    }
                    else if (status_p != 0) {
                        fprintf(results_file, "    Sample %d: Hord failed with perturbation (Status %d)\n", k + 1, status_p);
                        fprintf(fp, "%.2f\t%.16e\n", (double)p, NAN); // Indicate failure
                    }
                    else {
                        // Unperturbed root is near zero, relative error is tricky
                        fprintf(results_file, "    Sample %d: Unperturbed root near zero, using absolute error instead.\n", k + 1);
                        err_rel = fabs(root_perturbed - root_unperturbed); // Use absolute error as fallback
                        fprintf(fp, "%.2f\t%.16e\n", (double)p, err_rel); // Still log it
                    }
                } // End samples loop
            } // End percentage loop
            fclose(fp);
            fprintf(results_file, "  Perturbation data saved to %s\n", filename_perturb);
        } // End file open check
    } // End isnan check
    fprintf(results_file, "----------------------------------------\n\n");


    fprintf(results_file, "Analysis complete. Check generated .txt files and run plotter.py.\n");
    fclose(results_file);
    printf("C program finished. Results logged in results.txt. Data files generated.\n");

    return 0;
}