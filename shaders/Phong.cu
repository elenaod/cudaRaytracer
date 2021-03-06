#include <shaders/shading.cuh>

Phong::Phong(const Color& diffuse, double e, float str) :
  Shader(PHONG) {
  color = diffuse; exponent = e; strength = str;
}

Phong::Phong(const char* str) : Shader(PHONG) {
  sscanf(str, "%f%f%f%lf%f",
              &color.r, &color.g, &color.b,
              &exponent, &strength);
}
__host__ __device__
Color Phong::shade(const Ray& ray, const Light& light,
                   const bool& visibility,
                   const IntersectionData& data) const{
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
