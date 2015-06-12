#include <main/init.cuh>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
/*
  file follows the pattern:
  (geometry: [plane | sphere] <parameters>)*
  (shader: [checkered | shiny] <color>* )*
  (camera: yaw, pitch, roll, fov, aspect, pos.x, pos.y, pos.z)
  (lihttp://stackoverflow.com/questions/7146719/identifier-string-undefinedght: color.r, color.g, color.b, power, pos.x, pos.y, pos.z)
  (node: geometryIndex, shaderIndex)
*/

// this goes to utils
// ideally, find a parsing library 
bool getLineFrom(int f, char* line){
  char c; int i = 0;
  while( read(f, &c, 1) > 0 ){
    if (c == '\n'){
      line[i] = '\0';
      return true;
    }
    line[i] = c;
    ++i;
  }
  line[i] = '\0';
  return false;
}

// TODO: make better!
void Scene::readFromFile(const char* fileName){
  int sceneDesc = open(fileName, O_RDONLY);
  char l [1024];
  char object[10];

  while(getLineFrom(sceneDesc, l)){
    char *line = l;
    sscanf(line, "%s", object);
    line += strlen(object) + 1;
    if (strcmp(object, "geometry:") == 0) {
      char gtype[10];
      sscanf(line, "%s", gtype);
      printf("%s", gtype);
      line += strlen(gtype) + 1;
      if (strcmp(gtype, "plane") == 0) {
        int y;
        sscanf(line, "%d", &y);
        Plane *p = new Plane (y);
        Plane *dev_p = 0;
        cudaMalloc((void**)&dev_p, sizeof(Plane));
        cudaMemcpy(dev_p, p, sizeof(Plane), cudaMemcpyHostToDevice);
        geometries.push_back(dev_p);
        delete p;
      }
      else if (strcmp(gtype, "sphere") == 0){
        float x, y, z; double r;
        sscanf(line, "%f%f%f%lf", &x, &y, &z, &r);
        Sphere *s = new Sphere( Vector(x, y, z), r);
        Sphere *dev_s = 0;
        cudaMalloc((void**)&dev_s, sizeof(Sphere));
        cudaMemcpy(dev_s, s, sizeof(Sphere), cudaMemcpyHostToDevice);
        delete s;
        geometries.push_back(dev_s);
      }
    }
    else if (strcmp(object, "shader:") == 0) {
      char stype[10];
      sscanf(line, "%s", stype);
      line += strlen(stype) + 1;
      if (strcmp(stype, "checkered") == 0){
        Color c1, c2;
        double size;
        sscanf(line, "%f%f%f%f%f%f%lf",
                  &c1.r, &c1.g, &c1.b,
                  &c2.r, &c2.g, &c2.b, &size);
        CheckerShader *c = new CheckerShader(c1, c2, size);
        Shader *dev_c = 0;
        cudaMalloc((void**)&dev_c, sizeof(CheckerShader));
        cudaMemcpy(dev_c, c, sizeof(CheckerShader), cudaMemcpyHostToDevice);
        shaders.push_back(dev_c);
        delete c;
      }
      else if (strcmp(stype, "shiny") == 0){
        Color c1;
        double exponent; float strength;
        sscanf(line, "%f%f%f%lf%f",
                      &c1.r, &c1.g, &c1.b, &exponent, &strength);
        Phong *c = new Phong(Color(c1.r, c1.g, c1.b), exponent, strength);
        Phong *dev_c = 0;
        cudaMalloc((void**)&dev_c, sizeof(Phong));
        cudaMemcpy(dev_c, c, sizeof(Phong), cudaMemcpyHostToDevice);
        shaders.push_back(dev_c);
        delete c;
      }
    }
    else if (strcmp(object, "camera:") == 0){
      Camera *cam = new Camera;
      float aspectX, aspectY;
      sscanf(line, "%lf%lf%lf%lf%lf%lf%lf%f%f",
                    &cam->yaw, &cam->pitch, &cam->roll,
                    &cam->pos.x, &cam->pos.y, &cam->pos.z,
                    &cam->fov, &aspectX, &aspectY);
      cam->aspect = aspectX / aspectY;
      cam->beginFrame();
      cudaMalloc((void**)&camera, sizeof(Camera));
      cudaMemcpy(camera, cam, sizeof(Camera), cudaMemcpyHostToDevice);
      delete cam;
    }
    else if (strcmp(object, "light:") == 0){
      Light *_l = new Light;
      sscanf(line, "%f%f%f%lf%lf%lf%f",
                    &_l->color.r, &_l->color.g, &_l->color.b,
                    &_l->pos.x, &_l->pos.y, &_l->pos.z, &_l->power);
      cudaMalloc((void**)&light, sizeof(Light));
      cudaMemcpy(light, _l, sizeof(Light), cudaMemcpyHostToDevice);
      delete _l;
    }
    else if (strcmp(object, "node:") == 0){
      Node node;
      int geom_index, shader_index;

      sscanf(line, "%d%d", &geom_index, &shader_index);
      node.geom = (Geometry*) *(geometries.begin() + geom_index - 1);
      node.shader = (Shader*) *(shaders.begin() + shader_index - 1);

      nodes.push_back(node);
    }
    else return;
    memset(l, ' ', 1024);
  }
  start = nodes.begin(); end = nodes.end();
  printf("exiting readFroFile\n");
}

void Scene::initialize() {
  printf("BIG AND OBVIOUS!\n");
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
  host_light->color = Color(0.4, 0.0, 0.4);
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
