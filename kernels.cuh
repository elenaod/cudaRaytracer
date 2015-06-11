#include <init.cuh>

void setBuckets(const int& threadCount, const int& blocks);

__global__
void renderScene(Scene* scene, Color* buffer);

__global__
void findAA(bool* needsAA, Color* buffer);

__global__
void antialias(Scene* scene, bool* needsAA, Color* buffer);
