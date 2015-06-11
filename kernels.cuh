#include <init.cuh>

extern __constant__ int bucketSizeX, bucketSizeY,
                        bucketsX, bucketsY;

__global__
void renderScene(Scene* scene, Color* buffer);

__global__
void findAA(bool* needsAA, Color* buffer);

__global__
void antialias(Scene* scene, bool* needsAA, Color* buffer);
