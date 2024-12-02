#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

void generate_matrix(double** A, size_t n, double cond_number, int is_spd);
void perturb_matrix(double** A, size_t n, double perturbation, int perturb_max);
void multiply_matrix_vector(double** A, double* x, double* b, size_t n);
double calculate_condition_number(double** A, size_t n);
void copy_matrix(double** src, double** dest, size_t n);

#endif // MATRIX_H