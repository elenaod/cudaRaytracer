#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <basics/ray.cuh>
#include <basics/intersection.cuh>

enum geometry_type {
  GENERIC, PLANE, SPHERE
};

class Geometry {
  public:
    geometry_type t;
};

class Plane : public Geometry{
    int y;
  public:
    Plane() {y = 0; t = PLANE;}
    Plane(int _y) {y = _y; t = PLANE;}

    __host__ __device__
    bool intersect (const Ray& ray, IntersectionData& data);
};

class Sphere : public Geometry{
    Vector center;
    double R;
  public:
    Sphere() {t = SPHERE;}
    Sphere(const Vector& O, double r) : center(O), R(r) {t = SPHERE;}

    __host__ __device__
    bool intersect(const Ray& ray, IntersectionData& data);
};

#endif // __GEOMETRY_H__