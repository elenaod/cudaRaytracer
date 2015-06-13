#include <math_functions.h>
#include <geometries/geometry.cuh>
#include <utils/vector.cuh>
#include <utils/constants.cuh>
#include <cstdio>

Plane::Plane(int _y) :
   Geometry(PLANE) {
  y = _y;
}

Plane::Plane(const char* str) :
    Geometry(PLANE){
    sscanf(str, "%d", &y);
}

__host__ __device__
bool Plane::intersect(const Ray& ray, IntersectionData& data){
  if ((ray.start.y > y && ray.dir.y > -1e-9) ||
      (ray.start.y < y && ray.dir.y < 1e-9))
    return false;
  else {
    double yDiff = ray.dir.y;
    double wantYDiff = ray.start.y - this->y;
    double mult = wantYDiff / -yDiff;

    if (mult > data.dist) return false;

    // calculate intersection:
    data.p = ray.start + ray.dir * mult;
    data.dist = mult;
    data.normal = Vector(0, 1, 0);
    data.u = data.p.x;
    data.v = data.p.z;
    return true;
  }
}

