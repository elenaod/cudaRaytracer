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

    Geometry() {t = GENERIC;}
    Geometry(geometry_type type) {t = type;}
};

class Plane : public Geometry{
  public:
    int y;
    Plane(int _y);
    Plane(const char* str);

    __host__ __device__
    bool intersect (const Ray& ray, IntersectionData& data);
};

class Sphere : public Geometry{
    Vector center;
    double R;
  public:
    Sphere(const Vector& O, double r);
    Sphere(const char* str);

    __host__ __device__
    bool intersect(const Ray& ray, IntersectionData& data);
};

__device__
bool intersect(Geometry* geom, const Ray& ray, IntersectionData& data);
#endif // __GEOMETRY_H__
