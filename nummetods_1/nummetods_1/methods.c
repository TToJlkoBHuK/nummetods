#include <math.h>
#include "methods.h"

double bisection(FuncPtr f, double a, double b, double tol, int max_iter, int* iter_count, double* errors) {
    double fa = f(a);
    double fb = f(b);
    if (fa * fb >= 0) {
        return NAN;
    }

    double c, fc;
    *iter_count = 0;

    for (int i = 0; i < max_iter; i++) {
        c = (a + b) / 2.0;
        fc = f(c);
        errors[i] = fabs(fc);
        *iter_count += 1;

        if (fabs(fc) < tol || fabs(b - a) / 2.0 < tol) {
            return c;
        }

        if (fa * fc < 0) {
            b = c;
            fb = fc;
        }
        else {
            a = c;
            fa = fc;
        }
    }

    return c;
}

double secant(FuncPtr f, double x0, double x1, double tol, int max_iter, int* iter_count, double* errors) {
    double f0 = f(x0);
    double f1_val = f(x1);
    double x2, f2;
    *iter_count = 0;

    for (int i = 0; i < max_iter; i++) {
        if (fabs(f1_val - f0) < 1e-15) {
            return x1;
        }

        x2 = x1 - f1_val * (x1 - x0) / (f1_val - f0);
        f2 = f(x2);
        errors[i] = fabs(f2);
        *iter_count += 1;

        if (fabs(f2) < tol) {
            return x2;
        }

        x0 = x1;
        f0 = f1_val;
        x1 = x2;
        f1_val = f2;
    }

    return x1;
}