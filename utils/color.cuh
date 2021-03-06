#ifndef __COLOR_H__
#define __COLOR_H__

#include <utils/util.cuh>

/*
  Defines Color and implements arithmetic operations on it, conversions from one representation to another, and other helper functions, calculating intensity, difference between colors, etc.
*/

inline unsigned convertTo8bit(float x) {
  if (x < 0) x = 0;
  if (x > 1) x = 1;
  return nearestInt(x * 255.0f);
}

struct Color {
  float r, g, b;

  __host__ __device__
  Color() {}
  __host__ __device__
  Color(float _r, float _g, float _b) {
    setColor(_r, _g, _b);
  }
  __host__ __device__
  explicit Color(unsigned rgbcolor) {
    b = (rgbcolor & 0xff) / 255.0f;
    g = ((rgbcolor >> 8) & 0xff) / 255.0f;
    r = ((rgbcolor >> 16) & 0xff) / 255.0f;
  }
  // blue channel occupies least-significant byte
  unsigned toRGB32(int redShift = 16,
                   int greenShift = 8,
                   int blueShift = 0) {
    unsigned ir = convertTo8bit(r);
    unsigned ig = convertTo8bit(g);
    unsigned ib = convertTo8bit(b);
    return (ib << blueShift) | (ig << greenShift) | (ir << redShift);
  }

  __host__ __device__
  void makeBlack(void) {
    r = g = b = 0;
  }
  __host__ __device__
  void setColor(float _r, float _g, float _b) {
    r = _r; g = _g; b = _b;
  }

  // direct intensity
  __host__ __device__
  float intensity(void) {
    return (r + g + b) / 3;
  }
  // perceptual intensity
  __host__ __device__
  float intensityPerceptual(void) {
    return (r * 0.299 + g * 0.587 + b * 0.114);
  }

  __host__ __device__
  void operator += (const Color& rhs) {
    r += rhs.r; g += rhs.g; b += rhs.b;
  }
  __host__ __device__
  void operator *= (float coeff) {
    r *= coeff; g *= coeff; b *= coeff;
  }
  __host__ __device__
  void operator /= (float div)
  {
    r /= div; g /= div; b /= div;
  }
};

__host__ __device__
inline Color operator + (const Color& a, const Color& b) {
  return Color(a.r + b.r, a.g + b.g, a.b + b.b);
}

__host__ __device__
inline Color operator - (const Color& a, const Color& b) {
  return Color(a.r - b.r, a.g - b.g, a.b - b.b);
}

__host__ __device__
inline Color operator * (const Color& a, const Color& b) {
  return Color(a.r * b.r, a.g * b.g, a.b * b.b);
}

__host__ __device__
inline Color operator * (const Color& a, float coeff) {
  return Color(a.r * coeff, a.g * coeff, a.b * coeff);
}

__host__ __device__
inline Color operator * (float coeff, const Color& a) {
  return Color(a.r * coeff, a.g * coeff, a.b * coeff);
}

__host__ __device__
inline Color operator / (const Color& a, float div) {
  return Color(a.r / div, a.g / div, a.b / div);
}

__host__ __device__
inline bool tooDifferent(const Color& a, const Color& b)
{
  const float THRESHOLD = 0.1;
  return (fabs(a.r - b.r) > THRESHOLD ||
          fabs(a.g - b.g) > THRESHOLD ||
          fabs(a.b - b.b) > THRESHOLD);
}

inline void shiftColor(Color& c, int3& dirs, float3& shift){
    c += Color (shift.x*dirs.x, shift.y*dirs.y, shift.z*dirs.z);
    if (c.r > 1) { c.r = 1; dirs.x = -1;}
    else if (c.r < 0){ c.r = 0; dirs.x = 1;}
    if (c.b > 1) { c.b = 1; dirs.z = -1;}
    else if (c.b < 0){ c.b = 0; dirs.z = 1;}
    if (c.g > 1) { c.g = 1; dirs.y = -1;}
    else if (c.g < 0){ c.g = 0; dirs.y = 1;}
}
#endif // __COLOR_H__
