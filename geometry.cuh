#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include "ray.cuh"

struct IntersectionData {
  Vector p;
  Vector normal;
  double dist;

  double u, v;
};

enum geometry_type {
  GENERIC, PLANE
};

class Geometry {
  public:
    geometry_type t;

    Geometry() {}
    ~Geometry() {}
};

class Plane : public Geometry{
  public:
    int y;

    Plane() {y = 0; t = PLANE;}
    Plane(int _y) {y = _y; t = PLANE;}

    __host__ __device__
    bool intersect (Ray& ray, IntersectionData& data);
};
#endif // __GEOMETRY_H__
