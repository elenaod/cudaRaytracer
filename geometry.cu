#include "geometry.cuh"
#include "vector.cuh"
#include "constants.cuh"

#include <cstdio>
__host__ __device__
bool Plane::intersect(const Ray& ray, IntersectionData& data){
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

__host__ __device__
bool Sphere::intersect(const Ray& ray, IntersectionData& data){
  // compute the sphere intersection using a quadratic equation:
  Vector H = ray.start - center;

  double A = ray.dir.lengthSqr();
  double B = 2 * dot(H, ray.dir);
  double C = H.lengthSqr() - radius*radius;
  double dscr = B*B - 4*A*C;
  if (dscr < 0) return false; 

  double x1, x2;
  x1 = (-B + sqrt(dscr)) / (2*A);
  x2 = (-B - sqrt(dscr)) / (2*A);
  double sol = x2; // get the closer of the two solutions...
  sol = sol < 0 ? x1 : sol;
  if (sol < 0)
    return false;
  if (sol > data.dist)
    return false;

  data.dist = sol;
  data.p = ray.start + ray.dir * sol;
  data.normal = data.p - center; // generate the normal by getting the direction from the center to the ip
  data.normal.normalize();
  double angle = atan2(data.p.z - center.z, data.p.x - center.x);
  data.u = (PI + angle)/(2*PI);
  data.v = 1.0 - (PI/2 + asin((data.p.y - center.y)/radius)) / PI;
  return true;
}
