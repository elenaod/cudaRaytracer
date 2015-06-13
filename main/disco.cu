#include <SDL/SDL.h>
#include <utils/sdl.cuh>
#include <kernels.cuh>
#include <cstdio>

const int __SIZE = VFB_MAX_SIZE * VFB_MAX_SIZE;
SDL_Surface* screen = NULL;

void handleExit(bool& running){
  SDL_Event ev;

  while (SDL_PollEvent(&ev)) {
    switch (ev.type) {
      case SDL_QUIT:
        running = false;
        return;
    }
  }
}

int main(int argc, char** argv) {
  bool running = true;
  unsigned threadCount, numBlocks;

  if (argc < 2) {
    printf("You need to specify the number of threads\n");
    printf("raytracer [threadCount] [numBlocks], by default numBlocks == 1");
    return 1;
  }
  else {
    threadCount = atoi(argv[1]);
    if (argc >= 3)
      numBlocks = atoi(argv[2]);
    else numBlocks = 1;
  }
  
  setBuckets(threadCount, numBlocks);
  Color *host_vfb, *device_vfb;
  host_vfb = (Color*)malloc(__SIZE * sizeof(Color));
  cudaMalloc((void**)&device_vfb, __SIZE * sizeof(Color));
  cudaMemcpy(device_vfb, host_vfb, __SIZE * sizeof(Color),
             cudaMemcpyHostToDevice);

  bool *needsAA;
  cudaMalloc((void**)&needsAA, __SIZE * sizeof(bool));

  Scene *scene = new Scene, *dev_scene;
  int3 dirs; float3 shift; shift.x = 0.5; shift.y = 0.1; shift.z = 0.1;
  dirs.x = 1; dirs.y = 1; dirs.z = 1;
  if (!initGraphics(&screen, RESX, RESY)) return -1;
  scene->readFromFile("sampleScene.txt");
  printf("read file\n");
  cudaMalloc((void**)&dev_scene, sizeof(Scene));
  cudaMemcpy(dev_scene, scene, sizeof(Scene), cudaMemcpyHostToDevice);

  Light* host_light = new Light();
  cudaMemcpy(host_light, scene->light, sizeof(Light), cudaMemcpyDeviceToHost);
  while (running) {
    int lp = 1;
    renderScene<<<1, threadCount>>>(dev_scene, device_vfb);
    findAA<<<1, threadCount>>>(needsAA, device_vfb);
    antialias<<<1, threadCount>>>(dev_scene, needsAA, device_vfb);
    cudaError_t cudaerr = cudaDeviceSynchronize();

    cudaMemcpy(host_vfb, device_vfb, __SIZE * sizeof(Color),
               cudaMemcpyDeviceToHost);
    displayVFB(screen, host_vfb);
    shiftColor(host_light->color, dirs, shift);
    host_light->power += 1000 * lp;
    if (host_light->power > 100000) lp = -1;
    if (host_light->power < 20000) lp  = +1;
    cudaMemcpy(scene->light, host_light, sizeof(Light),
               cudaMemcpyHostToDevice);
    handleExit(running);
  }

  closeGraphics();

  scene->cleanUp();
  free(scene); cudaFree(dev_scene);
  free(host_vfb);
  cudaFree(device_vfb);
  cudaFree(needsAA);

  return 0;
}
