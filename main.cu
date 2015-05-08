#include <SDL/SDL.h>
#include <vector>
#include "sdl.cuh"
#include "matrix.cuh"
#include "camera.cuh"
#include "geometry.cuh"
#include "shading.cuh"
using namespace std;

#include <cstdio>

// remove global variables
// come to think of it, separate struct for the scene?
// I need the camera and the nodes everywhere...

Color raytrace(Ray ray, const Light& _light, const vector<Node*>& _nodes){
  IntersectionData data;
  for (int i = 0; i < (int) _nodes.size(); i++)
    if (_nodes[i]->geom->intersect(ray, data))
      return _nodes[i]->shader->shade(ray, _light, data);

  return Color(0, 0, 0);
}

// makes scene == camera + geometries + shaders + lights
void initializeScene(Camera*& _camera,
                     Light* _light,
                     vector<Geometry*>& _geometries,
                     vector<Shader*>& _shaders,
                     vector<Node*>& _nodes) {
  _camera = new Camera;
  _camera->yaw = 0;
  _camera->pitch = -30;
  _camera->roll = 0;
  _camera->fov = 90;
  _camera->aspect = 4. / 3.0;
  _camera->pos = Vector(0,165,0);

  _camera->beginFrame();

  // so these go where? - we need them in shaders and cameras
  // separate struct for the light so I don't pass a ton of args
  _light->pos = Vector(-30, 100, 250);
  _light->color = Color(1, 1, 1);
  _light->power = 50000;

  Plane* plane = new Plane(2);
  _geometries.push_back(plane);

  CheckerShader* checker =
             new CheckerShader(Color(0, 0, 0), Color(0, 0.5, 1), 5);
  Node* floor = new Node(plane, checker);
  _shaders.push_back(checker);
  _nodes.push_back(floor);
}

void renderScene(const Camera& _camera,
                 const Light& _light,
                 const vector<Node*>& _nodes,
                 Color** buffer) {
  for (int y = 0; y < RESY; y++)
    for (int x = 0; x < RESX; x++) {
      Ray ray = _camera.getScreenRay(x, y);
      buffer[y][x] = raytrace(ray, _light, _nodes);
    }
}

int main(int argc, char** argv) {
  Color** vfb;
  SDL_Surface* screen = NULL;
  vfb = new Color* [VFB_MAX_SIZE];
  for(int i = 0; i < VFB_MAX_SIZE; ++i)
    vfb[i] = new Color [VFB_MAX_SIZE];
  printf("Program start...\n");

  Camera *camera = 0;
  Light pointLight;
  vector<Geometry*> geometries;
  vector<Shader*> shaders;
  vector<Node*> nodes;

  printf("Variables declared...\n");
  if (!initGraphics(&screen, RESX, RESY)) return -1;
  printf("Graphics initialized...\n");
  initializeScene(camera, &pointLight, geometries, shaders, nodes);
  printf("Scene initialized... camera = %d\n", camera);
  printf("Scene initialized... light color = (%f, %f, %f)\n",
      pointLight.color.r, pointLight.color.g, pointLight.color.b);
  printf("Scene initialized... light power = %f\n", pointLight.power);
  renderScene(*camera, pointLight, nodes, vfb);
  displayVFB(screen, vfb);
  // remove so we can time
  waitForUserExit();

  closeGraphics();
  return 0;
}
