#include <math.h>
#include "functions.h"

double f1(double x) {
    return 0.1 * x * x - x * log(x);
}

double f2(double x) {
    return x * x * x * x - 3 * x * x + 75 * x - 9999;
}

// x=2
double f3(double x) {
    if (fabs(x - 2.0) < 1e-8) {
        return 1e8;
    }
    else {
        return (x - 2.0) * (x + 3.0);
    }
}