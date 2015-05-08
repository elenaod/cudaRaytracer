#ifndef __COLOR_H__
#define __COLOR_H__

// why use two classes for color and vector?!
// can't color be some sort of vector?
#include "util.cuh"

inline unsigned convertTo8bit(float x) {
  if (x < 0) x = 0;
  if (x > 1) x = 1;
  return nearestInt(x * 255.0f);
}

struct Color {
  float r, g, b;

  Color() {}
  Color(float _r, float _g, float _b) {
    setColor(_r, _g, _b);
  }
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

  void makeBlack(void) {
    r = g = b = 0;
  }
  void setColor(float _r, float _g, float _b) {
    r = _r; g = _g; b = _b;
  }

  // direct intensity
  float intensity(void) {
    return (r + g + b) / 3;
  }
  // perceptual intensity
  float intensityPerceptual(void) {
    return (r * 0.299 + g * 0.587 + b * 0.114);
  }

  void operator += (const Color& rhs) {
    r += rhs.r; g += rhs.g; b += rhs.b;
  }
  void operator *= (float coeff) {
    r *= coeff; g *= coeff; b *= coeff;
  }
  void operator /= (float div)
  {
    r /= div; g /= div; b /= div;
  }
};

inline Color operator + (const Color& a, const Color& b) {
  return Color(a.r + b.r, a.g + b.g, a.b + b.b);
}

inline Color operator - (const Color& a, const Color& b) {
  return Color(a.r - b.r, a.g - b.g, a.b - b.b);
}

inline Color operator * (const Color& a, const Color& b) {
  return Color(a.r * b.r, a.g * b.g, a.b * b.b);
}

inline Color operator * (const Color& a, float coeff) {
  return Color(a.r * coeff, a.g * coeff, a.b * coeff);
}

inline Color operator * (float coeff, const Color& a) {
  return Color(a.r * coeff, a.g * coeff, a.b * coeff);
}

inline Color operator / (const Color& a, float div) {
  return Color(a.r / div, a.g / div, a.b / div);
}

#endif // __COLOR_H__
