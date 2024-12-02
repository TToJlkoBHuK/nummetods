#ifndef JACOBI_H
#define JACOBI_H

#include <stddef.h>

int jacobi_method(double** A, double* b, double* x, size_t n, double tol, int max_iter, double* errors);

#endif // JACOBI_H