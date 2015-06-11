#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdlib.h>
#include <math.h>
#include "constants.cuh"

__device__
inline double signOf(double x) { return x > 0 ? +1 : -1; }
__device__
inline double sqr(double a) { return a * a; }
__host__ __device__
inline double toRadians(double angle) { return angle / 180.0 * PI; }
__device__
inline double toDegrees(double angle_rad) { return angle_rad / PI * 180.0; }
inline int nearestInt(float x) { return (int) floor(x + 0.5f); }

// implement better random algorithm
__device__
inline float randomFloat() { return rand() / (float) RAND_MAX; }

#endif // __UTIL_H__
