#ifndef __RAY_H__
#define __RAY_H__

#include "vector.cuh"

struct Ray {
  Vector start, dir; // dir should be normalized
                     // why not do it here?!
  Ray() {}
  Ray(const Vector& _start, const Vector& _dir) {
    start = _start;
    dir = _dir;
  }
};

#endif // __RAY_H__
