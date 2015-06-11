#ifndef __VECTOR3D_H__
#define __VECTOR3D_H__

#include <math.h>

struct Vector {
  double x, y, z;

  __host__ __device__
  Vector () {}
  __host__ __device__
  Vector(double _x, double _y, double _z) { set(_x, _y, _z); }

  __host__ __device__
  void set(double _x, double _y, double _z) {
    x = _x; y = _y; z = _z;
  }
  __host__ __device__
  void makeZero(void) { x = y = z = 0.0; }

  __host__ __device__
  inline double length(void) const{
    return sqrt(x * x + y * y + z * z);
  }

  __host__ __device__
  inline double lengthSqr(void) const {
    return (x * x + y * y + z * z);
  }

  __host__ __device__
  void scale(double coeff) {
    x *= coeff; y *= coeff; z *= coeff;
  }
  __host__ __device__
  void operator *= (double coeff) {
    scale(coeff);
  }
  __host__ __device__
  void operator += (const Vector& rhs) {
    x += rhs.x; y += rhs.y; z += rhs.z;
  }
  __host__ __device__
  void operator /= (double div) {
    scale(1.0 / div);
  }
  __host__ __device__
  void normalize(void) {
    double coeff = 1.0 / length();
    scale(coeff);
  }

  __host__ __device__
  void setLength(double newLength) {
    scale(newLength / length());
  }
};

__host__ __device__
inline Vector operator + (const Vector& a, const Vector& b) {
  return Vector(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline Vector operator - (const Vector& a, const Vector& b) {
  return Vector(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__
inline Vector operator - (const Vector& a) {
  return Vector(-a.x, -a.y, -a.z);
}

__host__ __device__
inline double operator * (const Vector& a, const Vector& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__
inline double dot(const Vector& a, const Vector& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
inline Vector operator ^ (const Vector& a, const Vector& b) {
  return Vector(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

__host__ __device__
inline Vector operator * (const Vector& a, double coeff) {
  return Vector(a.x * coeff, a.y * coeff, a.z * coeff);
}
__host__ __device__
inline Vector operator * (double coeff, const Vector& a) {
  return Vector(a.x * coeff, a.y * coeff, a.z * coeff);
}
__host__ __device__
inline Vector operator / (const Vector& a, double div) {
  double coeff = 1.0 / div;
  return Vector(a.x * coeff, a.y * coeff, a.z * coeff);
}

__host__ __device__
inline Vector reflect(const Vector& ray, const Vector& norm)
{
  Vector result = ray - 2 * dot(ray, norm) * norm;
  result.normalize();
  return result;
}

__host__ __device__
inline Vector faceforward(const Vector& ray, const Vector& norm)
{
  if (ray * norm < 0) return norm;
  else return -norm;
}
#endif // __VECTOR3D_H__
