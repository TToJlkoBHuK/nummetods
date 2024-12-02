#ifndef LU_H
#define LU_H

#include <stddef.h>

void lu_decomposition(double** A, double** L, double** U, size_t n);
void lu_solve(double** L, double** U, double* b, double* x, size_t n);

#endif // LU_H