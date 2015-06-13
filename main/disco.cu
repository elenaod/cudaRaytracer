#include <SDL/SDL.h>
#include <utils/sdl.cuh>
#include <main/kernels.cuh>
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

bool getCommandLineArgs(int argc, char** argv,
                        unsigned& threadCount, unsigned& numBuckets,
                        unsigned& resX, unsigned& resY,
                        char* inputFile, char* outputFile) {
  unsigned i = 1;
  bool fail = true;

  printf("%d", argc);
  for(; i < argc;){
    if (strcmp(argv[i], "--threadCount")){
      threadCount = atoi(argv[i + 1]); i += 1; fail = false;
    }
    else if (strcmp(argv[i], "--inputFile")){
      strcpy(inputFile, argv[i + 1]); i += 2; fail = false;
    }
    else if (strcmp(argv[i], "--numBuckets")){
      numBuckets = atoi(argv[i + 1]); i += 2;
    }
    else if (strcmp(argv[i], "--outputFile")){
      strcpy(outputFile, argv[i + 1]); i += 2;
    }
    else if (strcmp(argv[i], "--resX")){
      resX = atoi(argv[i + 1]); i += 2;
    }
    else if (strcmp(argv[i], "--resY")){
      resY = atoi(argv[i + 1]); i += 2;
    }
    else {
      printf("Unknown argument!"); fail = true;
      return fail;
    }
  }

  return fail;
}

int main(int argc, char* argv[]) {
  bool running = true;
  unsigned threadCount, numBuckets = 0, resX = 0, resY = 0;
  char inputFile[1024], outputFile[1024]; 

  printf("Starting program...");
  bool args = getCommandLineArgs(argc, argv,
                                 threadCount, numBuckets,
                                 resX, resY, inputFile, outputFile);
  if (!args) {
    printf("command line options are:\n");
    printf("--threadCount, compulsory\n");
    printf("--inputFile, compulsory, for an example see main/sampleScene.txt\n");
    printf("--outputFile, optional, location to save the file");
    printf("--resX, --resY - resolution, optional");
  }
  else {
    printf("read args:\n");
    printf("threadCount: %d\n", threadCount);
    if (numBuckets == 0) numBuckets = 1;
    if (resX == 0) resX = 640;
    if (resY == 0) resY = 480;
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
  int3 dirs; float3 shift; shift.x = 0.5; shift.y = 0.1; shift.z = 0.1;
  dirs.x = 1; dirs.y = 1; dirs.z = 1;

  if (!initGraphics(&screen, resX, resY)) return -1;
  scene->readFromFile("main/sampleScene.txt");
  printf("read file!");
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
