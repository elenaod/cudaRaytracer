#include <SDL/SDL.h>
#include "sdl.cuh"
#include "init.cuh"
#include <cstdio>


typedef thrust::device_vector<Node>::iterator iterator;
typedef thrust::device_vector<Geometry*>::iterator geom_iterator;
typedef thrust::device_vector<Shader*>::iterator shader_iterator;

SDL_Surface* screen = NULL;
/*
  those are read-only device variables which are set in main depending on the input data (screen res, etc), and are used in renderScene
*/
__constant__ int bucketSizeX = 32, bucketSizeY = 24,
                 bucketsX = 5, bucketsY = 5;

__device__
bool intersect(Geometry* geom, const Ray& ray, IntersectionData& data){
  switch(geom->t){
    case PLANE: {
      Plane *p = (Plane*) geom;
      return p->intersect(ray, data);
    }
    case SPHERE: {
      Sphere *s = (Sphere*) geom;
      return s->intersect(ray, data);
    }
  };
  return 0;
}

__device__
Color shade(Shader* shader, const Ray& ray, const Light& light,
            const bool& visibility, const IntersectionData& data){
  switch(shader->t){
    case CHECKER: {
      CheckerShader* s = (CheckerShader*) shader;
      return s->shade(ray, light, data);
    }
    case PHONG: {
      Phong *ph = (Phong*) shader;
      return ph->shade(ray, light, visibility, data);
    }
  }
}

__device__
bool testVisibility(iterator start, iterator end, 
                    const Vector& from, const Vector& to){
  Ray ray;
  ray.start = from;
  ray.dir = to - from;
  ray.dir.normalize();

  IntersectionData temp;
  temp.dist = (to - from).length();

  for (iterator iter = start; iter != end; ++iter){
    Node value = *iter;
    if (intersect(value.geom, ray, temp))
      return false;
  }

  return true;
}

__device__
Color raytrace(const Ray& ray, const Light& _light,
               iterator start, iterator end){
  IntersectionData data;
  Shader* shader = 0;

  data.dist = 1e99;
  for (iterator iter = start; iter != end; ++iter){
    Node value = *iter;

    if(intersect(value.geom, ray, data)){
      shader = value.shader;
    }
  }

  if (shader != 0) {
    Vector normal = faceforward(ray.dir, data.normal);
    bool visibility = testVisibility(start, end, 
                        data.p + normal * 1e-6, _light.pos);
    return shade(shader, ray, _light, !visibility, data);
  }
  return Color(0, 0, 0);
}

__global__
void renderScene(const Camera* _camera, const Light* _light,
                 iterator start, iterator end,
                 Color* buffer) {
  // calculate thread idx
  int idx_thrd_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_thrd_y = blockIdx.y * blockDim.y + threadIdx.y;
  int grid_width = gridDim.x * blockDim.x;
  int idx_thread = idx_thrd_y * grid_width + idx_thrd_x;

  iterator s = start + 1;
  Node value = *(s);
  // calculate coordinates of pixel we're painting
  // remove constants
  int x = idx_thread / bucketsX;
  int y = idx_thread % bucketsY;
  for(int i = x * bucketSizeX; i < (x + 1) * bucketSizeX; ++i)
    for(int j = y * bucketSizeY; j < (y + 1) * bucketSizeY; ++j){
    Ray ray = _camera->getScreenRay(i, j);
    buffer[j * VFB_MAX_SIZE + i] = raytrace(ray, *_light, start, end);
  }
}

/*
  Possibly split this into two other functions for the 2nd and 3rd pass respectively (each pass should be a separate function)
  components: findAA() - 2nd pass, antialias() - 3rd pass;
*/
__global__
void findAA(bool* needsAA, Color* buffer){
  // calculate thread idx
  int idx_thrd_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_thrd_y = blockIdx.y * blockDim.y + threadIdx.y;
  int grid_width = gridDim.x * blockDim.x;
  int idx_thread = idx_thrd_y * grid_width + idx_thrd_x;

  // calculate coordinates of pixel we're painting
  // remove constants
  int x = idx_thread / bucketsX;
  int y = idx_thread % bucketsY;

  for (int i = x * bucketSizeX; i < (x + 1) * bucketSizeX; ++i) {
    for (int j = y * bucketSizeY; j < (y + 1) * bucketSizeY; ++j) {
      Color neighs[5];
      neighs[0] = buffer[j * VFB_MAX_SIZE + i];

      neighs[1] = buffer[j * VFB_MAX_SIZE + (i > 0 ? i - 1 : i)];
      neighs[2] = buffer[j * VFB_MAX_SIZE + 
                  (i + 1 < (x + 1) * bucketSizeX ? i + 1 : i)];

      neighs[3] = buffer[(j > 0 ? j - 1 : j) * VFB_MAX_SIZE + i];
      neighs[4] = buffer[
          (j + 1 < (y + 1) * bucketSizeY ? j + 1 : j) *VFB_MAX_SIZE + i];

      Color average(0, 0, 0);

      for (int k = 0; k < 5; k++)
        average += neighs[k];
      average /= 5.0f;

      for (int k = 0; k < 5; k++) {
        if (tooDifferent(neighs[k], average)) {
          needsAA[j * VFB_MAX_SIZE + i] = true;
          break;
        }
      }
    }
  }
}

__global__
void antialias(const Camera* _camera, const Light* _light,
               iterator start, iterator end, bool* needsAA, Color* buffer){
  // calculate thread idx
  int idx_thrd_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_thrd_y = blockIdx.y * blockDim.y + threadIdx.y;
  int grid_width = gridDim.x * blockDim.x;
  int idx_thread = idx_thrd_y * grid_width + idx_thrd_x;

  // calculate coordinates of pixel we're painting
  // remove constants
  int x = idx_thread / bucketsX;
  int y = idx_thread % bucketsY;

  const double kernel[5][2] = {
    { 0, 0 },
    { 0.3, 0.3 },
    { 0.6, 0 },
    { 0, 0.6 },
    { 0.6, 0.6 }};

  for (int i = x * bucketSizeX; i < (x + 1) * bucketSizeX; ++i) {
    for (int j = y * bucketSizeY; j < (y + 1) * bucketSizeY; ++j) {
      if (needsAA[j * VFB_MAX_SIZE + i]) {
        Color result = buffer[j * VFB_MAX_SIZE + i]; 
        for (int k = 1; k < 5; k++){
          Ray ray = _camera->getScreenRay(i + kernel[k][0], j + kernel[k][1]);
          result += raytrace(ray, *_light, start, end);
        }
        buffer[j * VFB_MAX_SIZE + i] = result / 5.0f;
      }
    }
  }
}


int main(int argc, char** argv) {
  clock_t init_start, init_end, draw_start, draw_end;
  float time;
  init_start = clock();

  const int __SIZE = VFB_MAX_SIZE * VFB_MAX_SIZE;
  Color *host_vfb, *device_vfb;
  bool *needsAA;
  host_vfb = (Color*)malloc(__SIZE * sizeof(Color));
  cudaMalloc((void**)&device_vfb, __SIZE * sizeof(Color));
  cudaMemcpy(device_vfb,
             host_vfb,
             __SIZE * sizeof(Color),
             cudaMemcpyHostToDevice);
  cudaMalloc((void**)&needsAA, __SIZE * sizeof(bool));

  Camera *camera = 0;
  Light *pointLight = 0;
  thrust::device_vector<Geometry*> geometries;
  thrust::device_vector<Shader*> shaders;
  thrust::device_vector<Node> nodes;

  if (!initGraphics(&screen, RESX, RESY)) return -1;
  initializeScene(camera, pointLight, geometries, shaders, nodes);

  iterator start = nodes.begin();
  iterator end = nodes.end();

  Node value = *start;
  CheckerShader *checker = new CheckerShader ();
  CheckerShader *dev_checker = (CheckerShader*) value.shader;
  printf("Found shader!");
  cudaMemcpy(checker, dev_checker,
             sizeof(CheckerShader), cudaMemcpyDeviceToHost);

  init_end = clock();
  time = ((float)init_end - (float)init_start) / CLOCKS_PER_SEC;
  printf("Sequential: %f s\n", time);
  while (true) {
    draw_start = clock();
    renderScene<<<1, 25>>>(camera, pointLight, start, end, device_vfb);
    findAA<<<1, 25>>>(needsAA, device_vfb);
    antialias<<<1, 25>>>(camera, pointLight, start, end, needsAA, device_vfb);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    draw_end = clock();
    time = ((float) draw_end - (float)draw_start) / CLOCKS_PER_SEC;
    printf("Parallel on N threads: %f s\n", time);

    cudaMemcpy(host_vfb,
              device_vfb,
               __SIZE * sizeof(Color),
               cudaMemcpyDeviceToHost);
    displayVFB(screen, host_vfb);
    checker->shiftColors();
    cudaMemcpy(dev_checker, checker,
               sizeof(CheckerShader), cudaMemcpyHostToDevice);
  }
  waitForUserExit();
  closeGraphics();
  printf("freeing memory!\n");

  // freeMemory() function:
  cudaFree(camera);
  cudaFree(pointLight);
  for(geom_iterator iter = geometries.begin();
      iter != geometries.end(); ++iter){
    Geometry* val = *iter;
    cudaFree(val);
  }
  for(shader_iterator iter = shaders.begin();
      iter != shaders.end(); ++iter){
    Shader* val = *iter;
    cudaFree(val);
  }
  free(host_vfb);
  cudaFree(device_vfb);
  cudaFree(needsAA);
  return 0;
}
