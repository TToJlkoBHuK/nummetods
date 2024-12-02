#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

double** allocate_matrix(size_t n);
void free_matrix(double** A, size_t n);
void random_orthogonal_matrix(double** A, size_t n);
void transpose_and_multiply(double** A, double** AtA, size_t n);
void invert_matrix(double** A, double** A_inv, size_t n);
double matrix_infinity_norm(double** A, size_t n);
double vector_infinity_norm(double* v1, double* v2, size_t n);

#endif // UTILS_H