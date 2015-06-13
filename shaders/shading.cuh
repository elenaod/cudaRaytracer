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
  shader_type t;

public:
  Shader() { t = GENERIC_SHADER; }
  Shader(shader_type type) { t = type; }

  __device__
  inline shader_type getType() const {return t;}

  ~Shader() {}
};

class CheckerShader: public Shader {
  Color color2;
public:
  double size;
  CheckerShader();
  CheckerShader(const Color& c1, const Color& c2, double size = 1);
  CheckerShader(const char* str);

  __host__ __device__
  Color shade(const Ray& ray,
              const Light& light,
              const IntersectionData& data) const;
};

class Phong: public Shader {
    double exponent;
    float strength;
  public:
    Phong(const Color& diffuse, double e = 16.0, float str = 1.0);
    Phong(const char* str);

    __host__ __device__
    Color shade(const Ray& ray,
                const Light& light,
                const bool& visibility,
                const IntersectionData& data) const;
};

#endif // __SHADING_H__
