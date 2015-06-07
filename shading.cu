#include "shading.cuh"
#include <cstdio>
Shader::Shader(const Color& color)
{
  this->color = color;
}

CheckerShader::CheckerShader(const Color& c1,
                             const Color& c2,
                             double size) : Shader(c1) {
  color2 = c2;
  this->size = size;
  t = CHECKER;
}

__host__ __device__
Color CheckerShader::shade(Ray& ray, const Light& light,
                           const IntersectionData& data) {
  // example - u = 150, -230
  // -> 1, -3
  int x = floor(data.u / size);
  int y = floor(data.v / size);
  int white = (x + y) % 2;

  Color result = white ? color2 : color;
  printf("%f, %f, %f\n", result.r, result.b, result.g);
//  result = result * light.color * light.power / 
//           (data.p - light.pos).lengthSqr();
//  Vector lightDir = light.pos - data.p;
//  lightDir.normalize();

//  double cosTheta = dot(lightDir, data.normal);
//  result = result * cosTheta;
  return result;
}
