#ifndef __SHADING_H__
#define __SHADING_H__

#include "color.cuh"
#include "ray.cuh"
#include "geometry.cuh"
#include "light.cuh"

// need the inheritance and the virtual methods
class Shader {
public:
  Color color;
  Shader(const Color& color);
  virtual ~Shader() {}

  __host__ __device__
  virtual Color shade(Ray& ray, const Light& light,
                      const IntersectionData& data) = 0;
};

class CheckerShader: public Shader {
public:

  Color color2;
  double size;
  CheckerShader(const Color& c1, const Color& c2, double size = 1);

  __host__ __device__
  Color shade(Ray& ray,
              const Light& light,
              const IntersectionData& data);
};

#endif // __SHADING_H__
