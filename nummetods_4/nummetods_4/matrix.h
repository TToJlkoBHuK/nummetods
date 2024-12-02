#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

void generate_matrix(double** A, size_t n, double cond_number, int is_spd, int determinant_case);
void multiply_matrix_vector(double** A, double* x, double* b, size_t n);
double calculate_condition_number(double** A, size_t n);
double calculate_determinant(double** A, size_t n);

#endif // MATRIX_H