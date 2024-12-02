#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

double** allocate_matrix(size_t n);
void free_matrix(double** A, size_t n);
void copy_matrix(double** src, double** dest, size_t n);
void transpose_matrix(double** A, double** At, size_t n);
void multiply_matrices(double** A, double** B, double** C, size_t n);
double matrix_norm(double** A, size_t n);
double vector_norm(double* v, size_t n);
double vector_norm_diff(double* v1, double* v2, size_t n);
void invert_matrix(double** A, double** A_inv, size_t n);

#endif // UTILS_H