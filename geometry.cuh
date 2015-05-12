#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include "ray.cuh"

struct IntersectionData {
  Vector p;
  Vector normal;
  double dist;

  double u, v;
};

// so... what to do with these?
class Geometry {
public:
  virtual ~Geometry() {}

__device__
  virtual bool intersect(Ray ray, IntersectionData& data) = 0;
};

class Plane: public Geometry {
  double y;
public:
  Plane(double _y) { y = _y; }

  __device__
  bool intersect(Ray ray, IntersectionData& data);
};

// could I possibly remove that?
class Shader;

class Node {
public:
  Geometry* geom;
  Shader* shader;

  Node() {}
  Node(Geometry* g, Shader* s) { geom = g; shader = s; }
};

#endif // __GEOMETRY_H__
