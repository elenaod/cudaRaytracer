#ifndef __SHADING_H__
#define __SHADING_H__

#include "color.cuh"
#include "ray.cuh"
#include "geometry.cuh"
#include "light.cuh"

// need the inheritance and the virtual methods

enum shader_type {
  GENERIC_SHADER, CHECKER, PHONG, LAMBERT
};

class Shader {
protected:
  Color color;
public:
  shader_type t;

  Shader() {}
  Shader(const Color& color);
  ~Shader() {}
};

class CheckerShader: public Shader {
  Color color2;
  double size;
  int direction;
public:
  CheckerShader() {t = CHECKER; direction = 1;}
  CheckerShader(const Color& c1, const Color& c2, double size = 1);

  __host__ __device__
  Color shade(const Ray& ray,
              const Light& light,
              const IntersectionData& data);
  void shiftColors();
};

class Phong: public Shader {
    double exponent;
    float strength;
  public:
    Phong() { t = PHONG; }
    Phong(const Color& diffuse, double e = 16.0, float str = 1.0);

    __host__ __device__
    Color shade(const Ray& ray,
                const Light& light,
                const bool& visibility,
                const IntersectionData& data);
};
#endif // __SHADING_H__
