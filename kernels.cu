#include <kernels.cuh>

__constant__ int bucketSizeX =128, bucketSizeY = 96,
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
void renderScene(Scene* scene, Color* buffer) {
  // calculate thread idx
  int idx_thrd_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_thrd_y = blockIdx.y * blockDim.y + threadIdx.y;
  int grid_width = gridDim.x * blockDim.x;
  int idx_thread = idx_thrd_y * grid_width + idx_thrd_x;

  // calculate coordinates of pixel we're painting
  // remove constants
  int x = idx_thread / bucketsX;
  int y = idx_thread % bucketsY;
  for(int i = x * bucketSizeX; i < (x + 1) * bucketSizeX; ++i)
    for(int j = y * bucketSizeY; j < (y + 1) * bucketSizeY; ++j){
    Ray ray = scene->camera->getScreenRay(i, j);
    buffer[j * VFB_MAX_SIZE + i] = raytrace(ray, *scene->light,
                                            scene->start, scene->end);
  }
}

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
void antialias(Scene* scene, bool* needsAA, Color* buffer){
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
          Ray ray = scene->camera->getScreenRay(i + kernel[k][0],
                                                j + kernel[k][1]);
          result += raytrace(ray, *scene->light, scene->start, scene->end);
        }
        buffer[j * VFB_MAX_SIZE + i] = result / 5.0f;
      }
    }
  }
}

