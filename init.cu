#include <init.cuh>
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
  while(read(f, &c, 1) > 0){
    if (c == '\n'){
      line[i] = '\0';
      return true;
    }
    line[i] = c;
    ++i;
  }
  return false;
}

// TODO: make better!
void Scene::readFromFile(const char* fileName){
  int sceneDesc = open(fileName, O_RDONLY);
  char l [1024];
  char object[10];

  while(getLineFrom(sceneDesc, l)){
    sscanf(l, "%s", object);
    printf("Read (%s) is %s", l, object);
    if (strcmp(object, "geometry:") == 0) {
      char gtype[10];
      sscanf(l, "%s%s", object, gtype);
      printf("%s", gtype);
      if (strcmp(gtype, "plane") == 0) {
        printf(" is geometry::plane\n");
        int y;
        sscanf(l, "%s%s%d",object, gtype, &y);
        printf(" y is %d\n", y);
        Plane *p = new Plane (y);
        Plane *dev_p = 0;
        cudaMalloc((void**)&dev_p, sizeof(Plane));
        cudaMemcpy(dev_p, p, sizeof(Plane), cudaMemcpyHostToDevice);
        printf("\ndev_p = %d\n", dev_p);
        geometries.push_back(dev_p);
        delete p;
      }
      else if (strcmp(gtype, "sphere") == 0){
        printf(" is geometry::sphere\n");
        float x, y, z, r;
        sscanf(l, "%s%s%f%f%f%f", object, gtype, &x, &y, &z, &r);
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
      sscanf(l, "%s%s",object, stype);
      if (strcmp(stype, "checkered") == 0){
        printf(" is shader::checker\n");
        Color c1, c2;
        int size;
        sscanf(l, "%s%s%f%f%f%f%f%f%d", object, stype,
                  &c1.r, &c1.g, &c1.b,
                  &c2.r, &c2.g, &c2.b, &size);
        CheckerShader *c = new CheckerShader(c1, c2, size);
        CheckerShader *dev_c = 0;
        cudaMalloc((void**)&dev_c, sizeof(CheckerShader));
        cudaMemcpy(dev_c, c, sizeof(CheckerShader), cudaMemcpyHostToDevice);
        shaders.push_back(dev_c);
        delete c;
        printf("done with checker!\n");
      }
      else if (strcmp(stype, "shiny") == 0){
        printf(" is shader::shiny\n");
        Color c1;
        double exponent; float strength;
        sscanf(l, "%s%s%f%f%f%lf%f", object, stype,
                   &c1.r, &c1.g, &c1.b, exponent, strength);
//        Phong *c = new Phong(Color(c1.r, c1.g, c1.b), exponent, strength);
//        Phong *dev_c = 0;
//        cudaMalloc((void**)&dev_c, sizeof(Phong));
//        cudaMemcpy(dev_c, c, sizeof(Phong), cudaMemcpyHostToDevice);
//        shaders.push_back(dev_c);
//        delete c;
      }
    }
    else if (strcmp(object, "camera:") == 0){
        printf(" is camera\n");
      Camera *cam = new Camera;
      float aspectX, aspectY;
      sscanf(l, "%s%lf%lf%lf%lf%lf%lf%lf%lf%lf", object,
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
        printf(" is light\n");
      Light *_l = new Light;
      sscanf(l, "%s%lf%lf%lf%lf%lf%lf", object,
             &_l->color.r, &_l->color.g, &_l->color.b,
             &_l->pos.x, &_l->pos.y, &_l->pos.z);
      cudaMalloc((void**)&light, sizeof(Light));
      cudaMemcpy(light, _l, sizeof(Light), cudaMemcpyHostToDevice);
      delete _l;
    }
    else if (strcmp(object, "node:") == 0){
        printf(" is node\n");
      Node node;
      int geom_index, shader_index;
      sscanf(l, "%s%d%d", object, &geom_index, &shader_index);
      printf("print: %s, %d, %d\n", object, geom_index, shader_index);
      node.geom = (Geometry*) *(geometries.begin() + geom_index);
      printf("%d\n", node.geom);
      node.shader = (Shader*) *(shaders.begin() + shader_index);
      printf("done with node");
    }
    else return;
  }
  memset(l, ' ', 1024);
}

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
