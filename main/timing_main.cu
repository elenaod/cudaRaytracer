#include <SDL/SDL.h>
#include <utils/sdl.cuh>
#include <kernels.cuh>
#include <cstdio>

/*
  Modified main which does not present its output.
  It is also timed more thoroughly than the other variants, so more conclusive data could be gathered from measurements.
  The script timeRaytracer.sh runs it for all possible threadCounts in a specified range
*/
const int __SIZE = VFB_MAX_SIZE * VFB_MAX_SIZE;
SDL_Surface* screen = NULL;

int main(int argc, char** argv) {
  clock_t init_start, init_end;
  clock_t draw_start, draw_end;
  clock_t free_start, free_end;

  float total, init, render, finish;
  init_start = clock();
  unsigned threadCount, numBuckets;

  if (argc < 2) {
    printf("You need to specify the number of threads\n");
    printf("raytracer threadCount [numBuckets] [fileName]\n");
    printf("By default numBlocks == 1\n");
  }
  else {
    threadCount = atoi(argv[1]);
    if (argc >= 3)
      numBuckets = atoi(argv[2]);
  }

  setBuckets(threadCount, numBuckets);
  Color *host_vfb, *device_vfb;
  host_vfb = (Color*)malloc(__SIZE * sizeof(Color));
  cudaMalloc((void**)&device_vfb, __SIZE * sizeof(Color));
  cudaMemcpy(device_vfb, host_vfb, __SIZE * sizeof(Color),
             cudaMemcpyHostToDevice);

  bool *needsAA;
  cudaMalloc((void**)&needsAA, __SIZE * sizeof(bool));

  Scene *scene = new Scene, *dev_scene;
  if (!initGraphics(&screen, RESX, RESY)) return -1;
  scene->readFromFile("sampleScene.txt");
  cudaMalloc((void**)&dev_scene, sizeof(Scene));
  cudaMemcpy(dev_scene, scene, sizeof(Scene), cudaMemcpyHostToDevice);

  init_end = clock();

  draw_start = clock();
  renderScene<<<1, threadCount>>>(dev_scene, device_vfb);
  findAA<<<1, threadCount>>>(needsAA, device_vfb);
  antialias<<<1, threadCount>>>(dev_scene, needsAA, device_vfb);
  cudaError_t cudaerr = cudaDeviceSynchronize();
  draw_end = clock();

  free_start = clock();
  cudaMemcpy(host_vfb, device_vfb, __SIZE * sizeof(Color),
             cudaMemcpyDeviceToHost);
  if (argc >= 4) {
    SDL_SaveBMP(screen, argv[3]);
  }
  closeGraphics();
  scene->cleanUp();
  free(scene); cudaFree(dev_scene);
  free(host_vfb);
  cudaFree(device_vfb);
  cudaFree(needsAA);
  free_end = clock();

  init = ((float)init_end - (float)init_start) / CLOCKS_PER_SEC;
  render = ((float)draw_end - (float)draw_start) / CLOCKS_PER_SEC;
  finish = ((float)free_end - (float)free_start) / CLOCKS_PER_SEC;
  total = init + render + finish;
  printf("==== %d, %d ========\n", threadCount, numBuckets);
  printf("Total: %f s\n", total);
  printf("Init time: %f s\n", init);
  printf("Render time: %f s\n", render);
  printf("Cleanup time: %f s\n", finish);
  printf("====================\n");
  return 0;
}
