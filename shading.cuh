#ifndef __SHADING_H__
#define __SHADING_H__

#include "color.cuh"
#include "ray.cuh"
#include "geometry.cuh"
#include "light.cuh"

// need the inheritance and the virtual methods
class Shader {
protected:
  Color color;
public:
  Shader(const Color& color);
  virtual ~Shader() {}

__device__
  virtual Color shade(Ray& ray, const Light& light,
                      const IntersectionData& data) = 0;
};

class CheckerShader: public Shader {
  Color color2;
  double size;
public:
  CheckerShader(const Color& c1, const Color& c2, double size = 1);

  __device__
  Color shade(Ray& ray,
              const Light& light,
              const IntersectionData& data);
};

#endif // __SHADING_H__
