#include <init.cuh>
/*
  Initializes a scene
  A scene consists of:
    - _geometries, the figures present
    - _shaders, the shaders used
*/
void Scene::initialize() {
  Camera *host_camera = new Camera;
  host_camera->yaw = 0;
  host_camera->pitch = -15;
  host_camera->roll = 0;
  host_camera->fov = 90;
  host_camera->aspect = 4. / 3.0;
  host_camera->pos = Vector(0,165,0);

  host_camera->beginFrame();
  cudaMalloc((void**)&camera, sizeof(Camera));
  cudaMemcpy(camera, host_camera,
             sizeof(Camera), cudaMemcpyHostToDevice);
  free(host_camera);

  Light* host_light = new Light;
  // -30, 100, 250
  host_light->pos = Vector(-30, 100, 250);
  host_light->color = Color(1, 1, 1);
  host_light->power = 50000;

  cudaMalloc((void**)&light, sizeof(Light));
  cudaMemcpy(light, host_light,
             sizeof(Light), cudaMemcpyHostToDevice);
  free(host_light);

  Plane *plane = new Plane(-5);
  Plane *dev_plane = 0;
  cudaMalloc((void**)&dev_plane, sizeof(Plane));
  cudaMemcpy(dev_plane, plane, sizeof(Plane), cudaMemcpyHostToDevice);
  free(plane);
  geometries.push_back(dev_plane);

//  Sphere *sphere = new Sphere(Vector(-15, 50, 250), 30.0);
//  Sphere *dev_sphere = 0;
//  cudaMalloc((void**)&dev_sphere, sizeof(Sphere));
//  cudaMemcpy(dev_sphere, sphere, sizeof(Sphere), cudaMemcpyHostToDevice);
//  free(sphere);
//  _geometries.push_back(dev_sphere);

  CheckerShader* checker = new CheckerShader(Color(0.2, 0.4, 0.2),
                                             Color(0.6, 0.2, 0.6), 50);
  CheckerShader* dev_checker = 0;
  cudaMalloc((void**)&dev_checker, sizeof(CheckerShader));
  cudaMemcpy(dev_checker, checker,
             sizeof(CheckerShader), 
             cudaMemcpyHostToDevice);
  shaders.push_back(dev_checker);
  free(checker);
//  Phong *phong = new Phong(Color(0, 0.5, 0));
//  Phong *dev_phong = 0;
//  cudaMalloc((void**)&dev_phong, sizeof(Phong));
//  cudaMemcpy(dev_phong, phong, sizeof(Phong), cudaMemcpyHostToDevice);
//  _shaders.push_back(dev_phong);

  Node floor;
  floor.geom = dev_plane; floor.shader = dev_checker;
  nodes.push_back(floor);
//  Node object;
//  object.geom = dev_sphere; object.shader = dev_checker;
//  _nodes.push_back(object);

  start = nodes.begin(); end = nodes.end();
}

void Scene::cleanUp(){
  cudaFree(camera);
  cudaFree(light);
  for(thrust::device_vector<Geometry*>::iterator iter = geometries.begin();
      iter != geometries.end(); ++iter){
    Geometry* val = *iter;
    cudaFree(val);
  }
  for(thrust::device_vector<Shader*>::iterator iter = shaders.begin();
      iter != shaders.end(); ++iter){
    Shader* val = *iter;
    cudaFree(val);
  }

}
