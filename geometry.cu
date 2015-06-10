#include <math_functions.h>
#include "geometry.cuh"
#include "vector.cuh"
#include "constants.cuh"

#include <cstdio>
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

__host__ __device__
bool Sphere::intersect(const Ray& ray, IntersectionData& data) {
  // compute the sphere intersection using a quadratic equation:
  Vector H = ray.start - center;

  double A = ray.dir.lengthSqr();
  double B = 2 * dot(H, ray.dir);
  double C = H.lengthSqr() - R*R;
  double Dscr = B*B - 4*A*C;
  if (Dscr < 0) return false; // no solutions to the quadratic equation - then we don't have an intersection.
  double x1, x2;
  x1 = (-B + sqrt(Dscr)) / (2*A);
  x2 = (-B - sqrt(Dscr)) / (2*A);
  double sol = x2; // get the closer of the two solutions...
  if (sol < 0) sol = x1; // ... but if it's behind us, opt for the other one
  if (sol < 0) return false; // ... still behind? Then the whole sphere is behind us - no intersection.

  // if the distance to the intersection doesn't optimize our current distance, bail out:
  if (sol > data.dist) return false;

  data.dist = sol;
  data.p = ray.start + ray.dir * sol;
  data.normal = data.p - center; 
  data.normal.normalize();
  data.u = (PI + atan2(data.p.z - center.z, data.p.x - center.x))/(2*PI);
  data.v = 1.0 - ( PI/2 + asin((data.p.y - center.y)/R) ) / PI;
  return true;
}

