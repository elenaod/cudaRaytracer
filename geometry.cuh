#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include "ray.cuh"

struct IntersectionData {
  Vector p;
  Vector normal;
  double dist;

  double u, v;
};

class Geometry {
  public:
    __host__ __device__
    virtual bool intersect(Ray ray, IntersectionData& data) = 0;
    virtual ~Geometry() {}
};

class Plane : public Geometry{
  public:
    int y;

    Plane() {y = 0;}
    Plane(int _y) {y = _y;}

    __host__ __device__
    bool intersect (Ray ray, IntersectionData& data);
};

// could I possibly remove that?
class Shader;

struct Node {
public:
  Geometry* geom;
  Shader* shader;

  Node() {}
  Node(Geometry* g, Shader* s) { geom = g; shader = s; }
  void setNode(Geometry *g, Shader *s) {geom = g; shader = s;}
};

#endif // __GEOMETRY_H__
