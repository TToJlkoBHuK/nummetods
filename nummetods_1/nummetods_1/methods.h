#ifndef METHODS_H
#define METHODS_H

typedef double (*FuncPtr)(double);

double bisection(FuncPtr f, double a, double b, double tol, int max_iter, int* iter_count, double* errors);
double secant(FuncPtr f, double x0, double x1, double tol, int max_iter, int* iter_count, double* errors);

#endif