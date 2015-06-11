#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "vector.cuh"

struct Matrix {
  double m[3][3];
  __host__ __device__
  Matrix() {}
  __host__ __device__
  Matrix(double diagonalElement) {
    m[0][0] = m[1][1] = m[2][2] = diagonalElement;
    m[0][1] = m[0][2] = 0.0;
    m[1][0] = m[1][2] = 0.0;
    m[2][0] = m[2][1] = 0.0;
  }
};

__host__ __device__
inline Vector operator * (const Vector& v, const Matrix& m) {
  return Vector(v.x * m.m[0][0] + v.y * m.m[1][0] + v.z * m.m[2][0],
                v.x * m.m[0][1] + v.y * m.m[1][1] + v.z * m.m[2][1],
                v.x * m.m[0][2] + v.y * m.m[1][2] + v.z * m.m[2][2]);
}

__host__ __device__
inline void operator *= (Vector& v, const Matrix& a) { v = v*a; }

// matrix multiplication; result = a*b
__host__ __device__
Matrix operator * (const Matrix& a, const Matrix& b);

// finds the inverse of a matrix (assuming it exists)
__device__
Matrix inverseMatrix(const Matrix& a);

// finds the determinant of a matrix
__device__
double determinant(const Matrix& a);

// functions to around matrix around axis:
__host__ __device__
Matrix rotX(double angle);
__host__ __device__
Matrix rotY(double angle);
__host__ __device__
Matrix rotZ(double angle);

#endif // __MATRIX_H__
