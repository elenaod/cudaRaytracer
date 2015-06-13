#ifndef __SHADING_H__
#define __SHADING_H__

#include <utils/color.cuh>
#include <basics/ray.cuh>
#include <basics/intersection.cuh>
#include <basics/light.cuh>

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
  Shader(const Color& color) {this->color = color;}
  ~Shader() {}
};

class CheckerShader: public Shader {
  Color color2;
public:
  double size;
  CheckerShader() {t = CHECKER;}
  CheckerShader(const Color& c1, const Color& c2, double size = 1);
  CheckerShader(const char* str) {};

  __host__ __device__
  Color shade(const Ray& ray,
              const Light& light,
              const IntersectionData& data);
};

class Phong: public Shader {
    double exponent;
    float strength;
  public:
    Phong() { t = PHONG; }
    Phong(const Color& diffuse, double e = 16.0, float str = 1.0) {};
    Phong(const char* str) {};
    __host__ __device__
    Color shade(const Ray& ray,
                const Light& light,
                const bool& visibility,
                const IntersectionData& data);
};

#endif // __SHADING_H__
