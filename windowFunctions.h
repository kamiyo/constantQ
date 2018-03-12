#pragma once

#include <cmath>

#include "typedefs.h"

inline u32 next_power2(u32 value) {
    return 1 << (u32)(1 + floor(log2(value - 1)));
}

double hamming(double alpha, double i, u32 n) {
    return (alpha - (1 - alpha) * cos(2 * PI * i / (double)(n - 2)));
}