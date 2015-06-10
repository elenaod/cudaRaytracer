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

Phong::Phong(const Color& diffuse, double e, float str) :
  Shader(diffuse), exponent(e), strength(str){
  t = PHONG;
}

__host__ __device__
Color Phong::shade(const Ray& ray, const Light& light,
                   const bool& visibility,
                   const IntersectionData& data){
  // turn the normal vector towards us (if needed):
  Vector N = faceforward(ray.dir, data.normal);

  Color diffuseColor = this->color;
//  if (texture) diffuseColor = texture->getTexColor(ray, data.u, data.v, N);

  Color lightContrib = Color (0.2, 0.2, 0.2);
  Color specular(0, 0, 0);

  if ( visibility ) {
    Vector lightDir = light.pos - data.p;
    lightDir.normalize();

    // get the Lambertian cosine of the angle between the geometry's normal and
    // the direction to the light. This will scale the lighting:
    double cosTheta = dot(lightDir, N);

    // baseLight is the light that "arrives" to the intersection point
    Color baseLight = light.color * light.power / 
                      (data.p - light.pos).lengthSqr();

    lightContrib += baseLight * cosTheta; // lambertian contribution

    // R = vector after the ray from the light towards the intersection point
    // is reflected at the intersection:
    Vector R = reflect(-lightDir, N);

    double cosGamma = dot(R, -ray.dir);
    if (cosGamma > 0)
      specular += baseLight * pow(cosGamma, exponent) * strength; // specular contribution
  }
  // specular is not multiplied by diffuseColor, since we want the specular hilights to be
  // independent on the material color. I.e., a blue ball has white hilights
  // (this is true for most materials, and false for some, e.g. gold)
  return diffuseColor * lightContrib + specular;
}


