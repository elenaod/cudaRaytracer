#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <math.h>
#include <utils/constants.cuh>

bool getLineFrom(int f, char* line);
bool getCommandLineArgs(int argc, char** argv,
                        unsigned& threadCount, unsigned& numBuckets,
                        unsigned& resX, unsigned& resY,
                        char* inputFile, char* outputFile);
cudaError_t moveToDevice(void* src, void** dest, size_t bytes,
                         bool freeOnHost);

__device__
inline double signOf(double x) { return x > 0 ? +1 : -1; }

__device__
inline double sqr(double a) { return a * a; }

__host__ __device__
inline double toRadians(double angle) { return angle / 180.0 * PI; }

__device__
inline double toDegrees(double angle_rad) { return angle_rad / PI * 180.0; }
inline int nearestInt(float x) { return (int) floor(x + 0.5f); }

__device__
inline float randomFloat() { return rand() / (float) RAND_MAX; }
#endif // __UTIL_H__
