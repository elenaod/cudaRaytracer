#include <shaders/shading.cuh>
#include <cstdio>

CheckerShader::CheckerShader(const Color& c1,
                             const Color& c2,
                             double size) : Shader(CHECKER) {
  color = c1; color2 = c2;
  this->size = size;
}

CheckerShader::CheckerShader(const char* str) :
  Shader(CHECKER) {
  sscanf(str, "%f%f%f%f%f%f%lf",
               &color.r, &color.g, &color.b,
               &color2.r, &color2.g, &color2.b, &size);
}

__host__ __device__
Color CheckerShader::shade(const Ray& ray, const Light& light,
                           const IntersectionData& data) const{
  // example - u = 150, -230
  // -> 1, -3
  int x = floor(data.u / size);
  int y = floor(data.v / size);
  int white = (x + y) % 2;

  Color result = white ? color2 : color;
  result = result * light.color * light.power / 
           (data.p - light.pos).lengthSqr();

  Vector lightDir = light.pos - data.p;
  lightDir.normalize();

  double cosTheta = dot(lightDir, data.normal);
  result = result * cosTheta;
  return result;
}
