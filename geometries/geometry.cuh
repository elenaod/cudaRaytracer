#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <basics/ray.cuh>
#include <basics/intersection.cuh>

enum geometry_type {
  GENERIC, PLANE, SPHERE
};

class Geometry {
    geometry_type t;
  public:
    Geometry() {t = GENERIC;}
    Geometry(geometry_type type) {t = type;}

    __device__
    inline geometry_type getType() const {return t;}
};

class Plane : public Geometry{
    int y;
  public:
    Plane(int _y);
    Plane(const char* str);

    __host__ __device__
    bool intersect (const Ray& ray, IntersectionData& data) const;
};

class Sphere : public Geometry{
    Vector center;
    double R;
  public:
    Sphere(const Vector& O, double r);
    Sphere(const char* str);

    __host__ __device__
    bool intersect(const Ray& ray, IntersectionData& data) const;
};

#endif // __GEOMETRY_H__
