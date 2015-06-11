#include <SDL/SDL.h>
#include <utils/sdl.cuh>
#include <kernels.cuh>
#include <cstdio>

SDL_Surface* screen = NULL;

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

  Scene *scene = new Scene, *dev_scene;
  if (!initGraphics(&screen, RESX, RESY)) return -1;
  scene->initialize();
  cudaMalloc((void**)&dev_scene, sizeof(Scene));
  cudaMemcpy(dev_scene, scene, sizeof(Scene), cudaMemcpyHostToDevice);

  init_end = clock();
  time = ((float)init_end - (float)init_start) / CLOCKS_PER_SEC;
  printf("Sequential: %f s\n", time);
  draw_start = clock();
  renderScene<<<1, 25>>>(dev_scene, device_vfb);
  findAA<<<1, 25>>>(needsAA, device_vfb);
  antialias<<<1, 25>>>(dev_scene, needsAA, device_vfb);
  cudaError_t cudaerr = cudaDeviceSynchronize();
  draw_end = clock();
  time = ((float) draw_end - (float)draw_start) / CLOCKS_PER_SEC;
  printf("Parallel on N threads: %f s\n", time);

  cudaMemcpy(host_vfb,
            device_vfb,
             __SIZE * sizeof(Color),
             cudaMemcpyDeviceToHost);
  displayVFB(screen, host_vfb);
  SDL_SaveBMP(screen, "output.bmp");
  waitForUserExit();
  closeGraphics();

  scene->cleanUp();
  free(scene); cudaFree(dev_scene);
  free(host_vfb);
  cudaFree(device_vfb);
  cudaFree(needsAA);

  return 0;
}
