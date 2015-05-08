#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdlib.h>
#include <math.h>
#include "constants.cuh"

inline double signOf(double x) { return x > 0 ? +1 : -1; }
inline double sqr(double a) { return a * a; }
inline double toRadians(double angle) { return angle / 180.0 * PI; }
inline double toDegrees(double angle_rad) { return angle_rad / PI * 180.0; }
inline int nearestInt(float x) { return (int) floor(x + 0.5f); }

/// returns a random floating-point number in [0..1).
/// This is not a very good implementation. A better method is to be employed soon.
inline float randomFloat() { return rand() / (float) RAND_MAX; }

#endif // __UTIL_H__
