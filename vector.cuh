#ifndef __VECTOR3D_H__
#define __VECTOR3D_H__

#include <math.h>

struct Vector {
  double x, y, z;

  Vector () {}
  Vector(double _x, double _y, double _z) { set(_x, _y, _z); }

  void set(double _x, double _y, double _z) {
    x = _x; y = _y; z = _z;
  }
  void makeZero(void) { x = y = z = 0.0; }

  inline double length(void) const{
    return sqrt(x * x + y * y + z * z);
  }
  inline double lengthSqr(void) const {
    return (x * x + y * y + z * z);
  }

  void scale(double coeff) {
    x *= coeff; y *= coeff; z *= coeff;
  }
  void operator *= (double coeff) {
    scale(coeff);
  }
  void operator += (const Vector& rhs) {
    x += rhs.x; y += rhs.y; z += rhs.z;
  }
  void operator /= (double div) {
    scale(1.0 / div);
  }
  void normalize(void) {
    double coeff = 1.0 / length();
    scale(coeff);
  }

  void setLength(double newLength) {
    scale(newLength / length());
  }
};

inline Vector operator + (const Vector& a, const Vector& b) {
  return Vector(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline Vector operator - (const Vector& a, const Vector& b) {
  return Vector(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline Vector operator - (const Vector& a) {
  return Vector(-a.x, -a.y, -a.z);
}

inline double operator * (const Vector& a, const Vector& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline double dot(const Vector& a, const Vector& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vector operator ^ (const Vector& a, const Vector& b) {
  return Vector(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

inline Vector operator * (const Vector& a, double coeff) {
  return Vector(a.x * coeff, a.y * coeff, a.z * coeff);
}
inline Vector operator * (double coeff, const Vector& a) {
  return Vector(a.x * coeff, a.y * coeff, a.z * coeff);
}
inline Vector operator / (const Vector& a, double div) {
  double coeff = 1.0 / div;
  return Vector(a.x * coeff, a.y * coeff, a.z * coeff);
}

#endif // __VECTOR3D_H__
