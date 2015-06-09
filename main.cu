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
__constant__ int bucketSize = 160, bucketsX = 4, bucketsY = 4;

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
}

__device__
Color shade(Shader* shader, const Ray& ray, const Light& light,
            const IntersectionData& data){
  switch(shader->t){
    case CHECKER: {
      CheckerShader* s = (CheckerShader*) shader;
      return s->shade(ray, light, data);
    }
    case PHONG: {
      Phong *ph = (Phong*) shader;
      return ph->shade(ray, light, data);
    }
  }
}

__device__
Color raytrace(const Ray& ray, const Light& _light,
               iterator start, iterator end){
  IntersectionData data;

  for (iterator iter = start; iter != end; ++iter){
    Node value = *iter;

    if(intersect(value.geom, ray, data)){
      return shade(value.shader, ray, _light, data);
    }
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

  // calculate coordinates of pixel we're painting
  // remove constants
  int x = idx_thread / bucketsX;
  int y = idx_thread % bucketsY;
  for(int i = x * bucketSize; i < (x + 1) * bucketSize; ++i)
    for(int j = y * bucketSize; j < (y + 1) * bucketSize; ++j){
    Ray ray = _camera->getScreenRay(i, j);
    buffer[j * VFB_MAX_SIZE + i] = raytrace(ray, *_light, start, end);
  }
}

int main(int argc, char** argv) {
  clock_t init_start, init_end, draw_start, draw_end;
  float time;
  init_start = clock();

  const int __SIZE = VFB_MAX_SIZE * VFB_MAX_SIZE;
  Color *host_vfb, *device_vfb;

  host_vfb = (Color*)malloc(__SIZE * sizeof(Color));
  cudaMalloc((void**)&device_vfb, __SIZE * sizeof(Color));
  cudaMemcpy(device_vfb,
             host_vfb,
             __SIZE * sizeof(Color),
             cudaMemcpyHostToDevice);

  Camera *camera = 0;
  Light *pointLight = 0;
  thrust::device_vector<Geometry*> geometries;
  thrust::device_vector<Shader*> shaders;
  thrust::device_vector<Node> nodes;

  if (!initGraphics(&screen, RESX, RESY)) return -1;
  initializeScene(camera, pointLight, geometries, shaders, nodes);

  iterator start = nodes.begin();
  iterator end = nodes.end();

  init_end = clock();
  time = ((float)init_end - (float)init_start) / CLOCKS_PER_SEC;
  printf("Sequential: %f s\n", time);
  draw_start = clock();
  renderScene<<<1, 16>>>(camera, pointLight,
                        start, end, device_vfb);
  cudaError_t cudaerr = cudaDeviceSynchronize();
  draw_end = clock();
  time = ((float) draw_end - (float)draw_start) / CLOCKS_PER_SEC;
  printf("Parallel on N threads: %f s\n", time);

  cudaMemcpy(host_vfb,
             device_vfb,
             __SIZE * sizeof(Color),
             cudaMemcpyDeviceToHost);
  displayVFB(screen, host_vfb);
  waitForUserExit();
  closeGraphics();

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
  return 0;
}
