#ifndef GIVENS_H
#define GIVENS_H

#include <stddef.h>

void givens_rotation(double** A, double* b, size_t n);
void back_substitution(double** R, double* b, double* x, size_t n);

#endif // GIVENS_H