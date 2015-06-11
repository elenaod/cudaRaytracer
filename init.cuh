#include <thrust/device_vector.h>
#include <utils/matrix.cuh>
#include <basics/camera.cuh>
#include <geometries/geometry.cuh>
#include <shaders/shading.cuh>

struct Node {
public:
  Geometry* geom;
  Shader* shader;
  double dist;

  Node() {}
  Node(Geometry* g, Shader* s) {
    geom = g;
    shader = s;
  }

  void setNode(Geometry *g, Shader *s) {
    geom = g;
    shader = s;
  }
};

typedef thrust::device_vector<Node>::iterator iterator;
typedef thrust::device_vector<Geometry*>::iterator geom_iterator;
typedef thrust::device_vector<Shader*>::iterator shader_iterator;

struct SceneObjects {
  Light *light;
  iterator start, end;
};

struct Scene {
  Camera* camera;
  Light* light;
  iterator start, end;

  thrust::device_vector<Node> nodes;

  thrust::device_vector<Geometry*> geometries;
  thrust::device_vector<Shader*> shaders;

  void readFromFile(const char* fileName);
  void initialize();
  void cleanUp();
};
