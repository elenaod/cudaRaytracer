#include "geometry.cuh"
#include "vector.cuh"

__device__
bool Plane::intersect(Ray ray, IntersectionData& data)
{
  if (ray.dir.y >= 0) return false;
  else {
    double yDiff = ray.dir.y;
    double wantYDiff = ray.start.y - this->y;
    double mult = wantYDiff / -yDiff;
    data.p = ray.start + ray.dir * mult;
    data.dist = mult;
    data.normal = Vector(0, 1, 0);
    data.u = data.p.x;
    data.v = data.p.z;
    return true;
  }
}
