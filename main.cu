#include <SDL/SDL.h>
#include <thrust/device_vector.h>
#include "sdl.cuh"
#include "matrix.cuh"
#include "camera.cuh"
#include "geometry.cuh"
#include "shading.cuh"
#include "cuPrintf.cu"
using namespace std;

#include <cstdio>

struct Node {
public:
  Geometry* geom;
  Shader* shader;

  Node() {}
  Node(Geometry* g, Shader* s) { geom = g; shader = s; }
  void setNode(Geometry *g, Shader *s) {geom = g; shader = s;}
};

__device__
Color raytrace(Ray ray,
               const Light& _light,
               thrust::device_vector<Node>::iterator start,
               thrust::device_vector<Node>::iterator end){
  IntersectionData data;

  Node value = *start;
  Plane *pl = (Plane*) value.geom;
  pl->intersect(ray, data);

  for (thrust::device_vector<Node>::iterator iter = start;
         iter != end; ++iter){
    Node value = *iter;
    bool intersect; Color shade;
    switch(value.geom->t){
      case PLANE: {
        Plane* p = (Plane*) value.geom;
        intersect = p->intersect(ray, data);
        break;
      }
    };
    if (intersect){
      switch(value.shader->t){
        case CHECKER: {
          CheckerShader* s = (CheckerShader*) value.shader;
          shade = s->shade(ray, _light, data);
          return shade;
        }
      }
    }
  }

  return Color(0, 0, 0);
}

// makes scene == camera + geometries + shaders + lights
void initializeScene(Camera*& _camera,
                     Light*& _light,
                     thrust::device_vector<Geometry*>& _geometries,
                     thrust::device_vector<Shader*>& _shaders,
                     thrust::device_vector<Node>& _nodes) {
  _camera = new Camera;
  _camera->yaw = 0;
  _camera->pitch = -30;
  _camera->roll = 0;
  _camera->fov = 90;
  _camera->aspect = 4. / 3.0;
  _camera->pos = Vector(0,165,0);

  _camera->beginFrame();

  _light = new Light;
  _light->pos = Vector(-30, 100, 250);
  _light->color = Color(1, 1, 1);
  _light->power = 50000;

  Plane* plane = new Plane(2);
  Plane *dev_plane = 0;
  cudaMalloc((void**)&dev_plane, sizeof(Plane));
  cudaMemcpy(dev_plane, plane, sizeof(Plane), cudaMemcpyHostToDevice);
  free(plane);
  _geometries.push_back(dev_plane);

  CheckerShader* checker = new CheckerShader(Color(1, 1, 1),
                                             Color(0, 0, 0), 50);
  CheckerShader* dev_checker = 0;
  cudaMalloc((void**)&dev_checker, sizeof(CheckerShader));
  cudaMemcpy(dev_checker, checker,
             sizeof(CheckerShader), 
             cudaMemcpyHostToDevice);
  free(checker);
  _shaders.push_back(dev_checker);

  Node floor;
  floor.geom = dev_plane; floor.shader = dev_checker;
  _nodes.push_back(floor);
}

__global__
void renderScene(Camera* _camera,
                 Light* _light,
                 thrust::device_vector<Node>::iterator start,
                 thrust::device_vector<Node>::iterator end,
                 Color* buffer) {
  // calculate thread idx
  int idx_thrd_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_thrd_y = blockIdx.y * blockDim.y + threadIdx.y;
  int grid_width = gridDim.x * blockDim.x;
  int idx_thread = idx_thrd_y * grid_width + idx_thrd_x;

  // calculate coordinates of pixel we're painting
  // remove constants
  int x = idx_thread / 4;
  int y = idx_thread % 4;
  for(int i = x * 160; i < (x + 1) * 160; ++i)
    for(int j = y * 160; j < (y + 1) * 160; ++j){
    Ray ray = _camera->getScreenRay(i, j);
    buffer[j * RESX + i] = raytrace(ray, *_light, start, end);
  }

  printf("renderScene::Scene rendered for (%d, %d)\n", x, y);
}

int main(int argc, char** argv) {
  const int __PIX = RESX;
  const int __SIZE = __PIX * __PIX;
  Color *host_vfb, *device_vfb;
  SDL_Surface* screen = NULL;

  // get vfb on host
  host_vfb = (Color*)malloc(__SIZE * sizeof(Color));
  for(int i = 0; i < __PIX; ++i){
    for (int j = 0; j < __PIX; ++j)
      host_vfb[i * __PIX + j] = Color (1, 0, 0);
  }

  // get vfb on device
  cudaMalloc((void**)&device_vfb, __SIZE * sizeof(Color));
  cudaMemcpy(device_vfb,
             host_vfb,
             __SIZE * sizeof(Color),
             cudaMemcpyHostToDevice);
  printf("Program start...\n");

  // now those are no the host, originally!
  Camera *camera = 0;
  Light *pointLight = 0;
  thrust::device_vector<Geometry*> geometries;
  thrust::device_vector<Shader*> shaders;
  thrust::device_vector<Node> nodes;

  printf("Variables declared...\n");
  if (!initGraphics(&screen, RESX, RESY)) return -1;
  printf("Graphics initialized...\n");
  initializeScene(camera, pointLight, geometries, shaders, nodes);
  printf("Scene initialized... camera = %d\n", camera);

  printf("Scene initialized... nodes.size: %llu\n", nodes.size());
  printf("Scene initialized... start - end = %llu\n",
           nodes.end() - nodes.begin());

  Camera *device_camera = 0;
  cudaMalloc((void**) &device_camera, sizeof(Camera));
  cudaMemcpy(device_camera, camera,
             sizeof(Camera), cudaMemcpyHostToDevice);
  Light *device_light = 0;
  cudaMalloc((void**) &device_light, sizeof(Light));
  cudaMemcpy(device_light, pointLight,
             sizeof(Light), cudaMemcpyHostToDevice);

  thrust::device_vector<Node>::iterator start = nodes.begin();
  thrust::device_vector<Node>::iterator end = nodes.end();

  printf("Scene initialized... start - end with vars = %llu\n",
           end - start);

  renderScene<<<1, 16>>>(device_camera, device_light,
                        start, end, device_vfb);

  cudaMemcpy(host_vfb,
             device_vfb,
             __SIZE * sizeof(Color),
             cudaMemcpyDeviceToHost);
  printf("Scene rendered... \n");
  displayVFB(screen, host_vfb);
  // remove so we can time
  waitForUserExit();
  printf("Closing graphics...\n");
  // illegal memory access was encountered
  closeGraphics();
  printf("All done, only destructors remain...\n");
  // aand, free!
  free(device_camera);
  delete camera;
  free(host_vfb);
  cudaFree(device_vfb);
  // well, lots more to free, technically!
  return 0;
}
