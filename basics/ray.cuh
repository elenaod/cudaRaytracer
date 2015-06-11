#ifndef __RAY_H__
#define __RAY_H__

#include <utils/vector.cuh>

struct Ray {
  Vector start, dir; // dir should be normalized
                     // why not do it here?!
__device__
  Ray() {}
__device__
  Ray(const Vector& _start, const Vector& _dir) {
    start = _start;
    dir = _dir;
  }
};

#endif // __RAY_H__
