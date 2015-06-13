#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <math.h>
#include <utils/constants.cuh>

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

// simple parsing function to read line from a file
// returns false on file end
inline bool getLineFrom(int f, char* line){
  char c; int i = 0;
  while( read(f, &c, 1) > 0 ){
    if (c == '\n'){ line[i] = '\0'; return true;}
    line[i++] = c;
  }
  line[i] = '\0';
  return false;
}

#endif // __UTIL_H__
