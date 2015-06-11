#include <shaders/shading.cuh>

CheckerShader::CheckerShader(const Color& c1,
                             const Color& c2,
                             double size) : Shader(c1) {
  color2 = c2;
  this->size = size;
  t = CHECKER;
  direction = 1;
}

__host__ __device__
Color CheckerShader::shade(const Ray& ray, const Light& light,
                           const IntersectionData& data) {
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

// well that's stupid; apparently there's Color::shift()
void CheckerShader::shiftColors() {
  if (direction == 1) {
    color.r = 1 - color.r; color.b = 1 - color.b; color.g = 1 - color.g;
  }
  if (direction == 0) {
    color2.r = 1 - color2.r; color2.b = 1 - color2.b; color2.g = 1 - color2.g;
  }
  direction = 1 - direction;
} 

