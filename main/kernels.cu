#include <main/kernels.cuh>
#include <cstdio>

__constant__ unsigned bucketSizeX, bucketSizeY;
__constant__ unsigned bucketsX, bucketsY;

void setBuckets(const int& threadCount, const int& blocks){
  int rows = blocks, columns = threadCount / blocks;
  int sizeX = RESX / rows, sizeY = RESY / columns;

  cudaMemcpyToSymbol(bucketsX, &rows, sizeof(unsigned));
  cudaMemcpyToSymbol(bucketsY, &columns, sizeof(unsigned));
  cudaMemcpyToSymbol(bucketSizeX, &sizeX, sizeof(unsigned));
  cudaMemcpyToSymbol(bucketSizeY, &sizeY, sizeof(unsigned));
}

__device__
bool intersect(Geometry* geom, const Ray& ray, IntersectionData& data){
  switch(geom->getType()){
    case PLANE: {
      Plane *p = (Plane*) geom;
      return p->intersect(ray, data);
    }
    case SPHERE: {
      Sphere *s = (Sphere*) geom;
      return s->intersect(ray, data);
    }
  };
  return false;
}

__device__
Color shade(Shader* shader, const Ray& ray, const Light& light,
            const bool& visibility, const IntersectionData& data){
  switch(shader->getType()){
    case CHECKER: {
      CheckerShader* s = (CheckerShader*) shader;
      Color c = s->shade(ray, light, data);
      return c;
    }
    case PHONG: {
      Phong *ph = (Phong*) shader;
      return ph->shade(ray, light, visibility, data);
    }
  }
  return Color(0, 1, 0);
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

  // there's a constant 'oo' in utils/constants... couldn't resist
  data.dist = oo;
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

__device__
int2 calculateCoordinates(){
  // calculate thread idx
  int idx_thrd_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_thrd_y = blockIdx.y * blockDim.y + threadIdx.y;
  int grid_width = gridDim.x * blockDim.x;
  int idx_thread = idx_thrd_y * grid_width + idx_thrd_x;

  // calculate coordinates of pixel we're painting
  // remove constants
  int x = idx_thread % bucketsX;
  int y = idx_thread / bucketsX;
  return make_int2(x, y);
}

__global__
void renderScene(Scene* scene, Color* buffer) {
  int2 c = calculateCoordinates();

//  printf("Rendering scene...\n");
  for(int i = c.x * bucketSizeX; i < (c.x + 1) * bucketSizeX; ++i)
    for(int j = c.y * bucketSizeY; j < (c.y + 1) * bucketSizeY; ++j){
    Ray ray = scene->camera->getScreenRay(i, j);
//    printf("Coloring VFB...\n");
//    printf("ray is: %f, %f, %f\n", ray.dir.x, ray.dir.y, ray.dir.z);
    buffer[j * VFB_MAX_SIZE + i] = raytrace(ray, *scene->light,
                                            scene->start, scene->end);
  }
}

__global__
void findAA(bool* needsAA, Color* buffer){
  int2 c = calculateCoordinates();

  for (int i = c.x * bucketSizeX; i < (c.x + 1) * bucketSizeX; ++i) {
    for (int j = c.y * bucketSizeY; j < (c.y + 1) * bucketSizeY; ++j) {
      Color neighs[5];
      neighs[0] = buffer[j * VFB_MAX_SIZE + i];

      neighs[1] = buffer[j * VFB_MAX_SIZE + (i > 0 ? i - 1 : i)];
      neighs[2] = buffer[j * VFB_MAX_SIZE + 
                  (i + 1 < (c.x + 1) * bucketSizeX ? i + 1 : i)];

      neighs[3] = buffer[(j > 0 ? j - 1 : j) * VFB_MAX_SIZE + i];
      neighs[4] = buffer[
          (j + 1 < (c.y + 1) * bucketSizeY ? j + 1 : j) *VFB_MAX_SIZE + i];

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
void antialias(Scene* scene, bool* needsAA, Color* buffer){
  int2 c = calculateCoordinates();
  const double kernel[5][2] = {
    { 0, 0 },
    { 0.3, 0.3 },
    { 0.6, 0 },
    { 0, 0.6 },
    { 0.6, 0.6 }};

  for (int i = c.x * bucketSizeX; i < (c.x + 1) * bucketSizeX; ++i) {
    for (int j = c.y * bucketSizeY; j < (c.y + 1) * bucketSizeY; ++j) {
      if (needsAA[j * VFB_MAX_SIZE + i]) {
        Color result = buffer[j * VFB_MAX_SIZE + i]; 
        for (int k = 1; k < 5; k++){
          Ray ray = scene->camera->getScreenRay(i + kernel[k][0],
                                                j + kernel[k][1]);
          result += raytrace(ray, *scene->light, scene->start, scene->end);
        }
        buffer[j * VFB_MAX_SIZE + i] = result / 5.0f;
      }
    }
  }
}

