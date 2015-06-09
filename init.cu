#include "init.cuh"
/*
  Initializes a scene
  A scene consists of:
    - _geometries, the figures present
    - _shaders, the shaders used
*/
void initializeScene(Camera*& _camera,
                     Light*& _light,
                     thrust::device_vector<Geometry*>& _geometries,
                     thrust::device_vector<Shader*>& _shaders,
                     thrust::device_vector<Node>& _nodes) {
  Camera *host_camera = new Camera;
  host_camera->yaw = 0;
  host_camera->pitch = -30;
  host_camera->roll = 0;
  host_camera->fov = 90;
  host_camera->aspect = 4. / 3.0;
  host_camera->pos = Vector(0,165,0);

  host_camera->beginFrame();
  cudaMalloc((void**)&_camera, sizeof(Camera));
  cudaMemcpy(_camera, host_camera,
             sizeof(Camera), cudaMemcpyHostToDevice);
  free(host_camera);

  Light* host_light = new Light;
  // -30, 100, 250
  host_light->pos = Vector(-30, 100, 250);
  host_light->color = Color(1, 1, 1);
  host_light->power = 50000;

  cudaMalloc((void**)&_light, sizeof(Light));
  cudaMemcpy(_light, host_light,
             sizeof(Light), cudaMemcpyHostToDevice);
  free(host_light);

  Plane *plane = new Plane(2);
  Plane *dev_plane = 0;
  cudaMalloc((void**)&dev_plane, sizeof(Plane));
  cudaMemcpy(dev_plane, plane, sizeof(Plane), cudaMemcpyHostToDevice);
  free(plane);
  _geometries.push_back(dev_plane);
  
  Sphere *sphere = new Sphere(Vector(0, 15, 200), 10.0);
  Sphere *dev_sphere = 0;
  cudaMalloc((void**)&dev_sphere, sizeof(Sphere));
  cudaMemcpy(dev_sphere, sphere, sizeof(Sphere), cudaMemcpyHostToDevice);
  free(sphere);
  _geometries.push_back(dev_sphere);

  CheckerShader* checker = new CheckerShader(Color(1, 1, 1),
                                             Color(0, 0, 0), 50);
  CheckerShader* dev_checker = 0;
  cudaMalloc((void**)&dev_checker, sizeof(CheckerShader));
  cudaMemcpy(dev_checker, checker,
             sizeof(CheckerShader), 
             cudaMemcpyHostToDevice);
  free(checker);
  Phong *phong = new Phong(Color(0, 1, 0));
  Phong *dev_phong = 0;
  cudaMalloc((void**)&dev_phong, sizeof(Phong));
  cudaMemcpy(dev_phong, phong, sizeof(Phong), cudaMemcpyHostToDevice);
  _shaders.push_back(dev_phong);

  Node floor;
  floor.geom = dev_plane; floor.shader = dev_phong;
  _nodes.push_back(floor);
  Node object;
  object.geom = dev_sphere; object.shader = dev_checker;
  _nodes.push_back(object);
}
