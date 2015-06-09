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

Phong::Phong(const Color& diffuse, double e, float str) :
  Shader(diffuse), exponent(e), strength(str){
  t = PHONG;
}

__host__ __device__
Color Phong::shade(const Ray& ray, const Light& light,
                   const IntersectionData& data){
  // turn the normal vector towards us (if needed):
  Vector N = faceforward(ray.dir, data.normal);

  Color diffuseColor = this->color;
//  if (texture) 
//    diffuseColor = texture->getTexColor(ray, data.u, data.v, N);

  Color lightContrib = ambientLight;
  Color specular(0, 0, 0);
	
	if (testVisibility(data.p + N * 1e-6, lightPos)) {
		Vector lightDir = lightPos - data.p;
		lightDir.normalize();
		
		// get the Lambertian cosine of the angle between the geometry's normal and
		// the direction to the light. This will scale the lighting:
		double cosTheta = dot(lightDir, N);

		// baseLight is the light that "arrives" to the intersection point
		Color baseLight = lightColor * lightPower / (data.p - lightPos).lengthSqr();
		
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


