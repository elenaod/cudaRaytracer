#include <SDL/SDL.h>
#include <utils/sdl.cuh>
#include <kernels.cuh>
#include <cstdio>

const int __SIZE = VFB_MAX_SIZE * VFB_MAX_SIZE;
SDL_Surface* screen = NULL;

int main(int argc, char** argv) {
  clock_t init_start, init_end, draw_start, draw_end;
  float time;
  unsigned threadCount, numBlocks;

  if (argc < 3) {
    printf("You need to specify the number of threads\n");
    printf("raytracer [threadCount] [numBlocks], by default numBlocks == 1");
  }
  else {
    threadCount = atoi(argv[1]);
    numBlocks = atoi(argv[2]);
  }
  
  setBuckets(threadCount, numBlocks);
  init_start = clock();
  Color *host_vfb, *device_vfb;
  host_vfb = (Color*)malloc(__SIZE * sizeof(Color));
  cudaMalloc((void**)&device_vfb, __SIZE * sizeof(Color));
  cudaMemcpy(device_vfb, host_vfb, __SIZE * sizeof(Color),
             cudaMemcpyHostToDevice);

  bool *needsAA;
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
  renderScene<<<1, threadCount>>>(dev_scene, device_vfb);
  findAA<<<1, threadCount>>>(needsAA, device_vfb);
  antialias<<<1, threadCount>>>(dev_scene, needsAA, device_vfb);
  cudaError_t cudaerr = cudaDeviceSynchronize();
  draw_end = clock();
  time = ((float) draw_end - (float)draw_start) / CLOCKS_PER_SEC;
  printf("Parallel on %d threads: %f s\n", threadCount, time);

  cudaMemcpy(host_vfb, device_vfb, __SIZE * sizeof(Color),
             cudaMemcpyDeviceToHost);
  displayVFB(screen, host_vfb);
  if (argc >= 4) {
    SDL_SaveBMP(screen, argv[3]);
  }

  waitForUserExit();

  closeGraphics();

  scene->cleanUp();
  free(scene); cudaFree(dev_scene);
  free(host_vfb);
  cudaFree(device_vfb);
  cudaFree(needsAA);

  return 0;
}
