#include <math_functions.h>
#include <geometries/geometry.cuh>
#include <utils/vector.cuh>
#include <utils/constants.cuh>

__host__ __device__
bool Sphere::intersect(const Ray& ray, IntersectionData& data) {
  Vector H = ray.start - center;

  double A = ray.dir.lengthSqr();
  double B = 2 * dot(H, ray.dir);
  double C = H.lengthSqr() - R*R;
  double Dscr = B*B - 4*A*C;
  if (Dscr < 0)
    return false;

  double x1, x2;
  x1 = (-B + sqrt(Dscr)) / (2*A);
  x2 = (-B - sqrt(Dscr)) / (2*A);
  double sol = x2; 
  if (sol < 0)
    sol = x1; 
  if (sol < 0)
    return false; 

  if (sol > data.dist)
    return false;

  data.dist = sol;
  data.p = ray.start + ray.dir * sol;
  data.normal = data.p - center; 
  data.normal.normalize();

  data.u = (PI + atan2(data.p.z - center.z, data.p.x - center.x)) / (2*PI);
  data.v = 1.0 - ( PI/2 + asin((data.p.y - center.y)/R) ) / PI;

  return true;
}

