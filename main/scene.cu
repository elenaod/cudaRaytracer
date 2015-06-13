#include <main/scene.cuh>
#include <utils/util.cuh>
/*
  file follows the pattern:
  (geometry: [plane | sphere] <parameters>)*
  (shader: [checkered | shiny] <color>* )*
  (camera: yaw, pitch, roll, fov, aspect, pos.x, pos.y, pos.z)
  (light: color.r, color.g, color.b, power, pos.x, pos.y, pos.z)
  (node: geometryIndex, shaderIndex)
*/

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
      line += strlen(gtype) + 1;
      if (strcmp(gtype, "plane") == 0) {
        Plane *p = new Plane (line);
        Plane *dev_p = 0;
        cudaMalloc((void**)&dev_p, sizeof(Plane));
        cudaMemcpy(dev_p, p, sizeof(Plane), cudaMemcpyHostToDevice);
        geometries.push_back(dev_p);
        delete p;
      }
      else if (strcmp(gtype, "sphere") == 0){
        Sphere *s = new Sphere(line);
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
        CheckerShader *c = new CheckerShader(line);
        Shader *dev_c = 0;
        cudaMalloc((void**)&dev_c, sizeof(CheckerShader));
        cudaMemcpy(dev_c, c, sizeof(CheckerShader), cudaMemcpyHostToDevice);
        shaders.push_back(dev_c);
        delete c;
      }
      else if (strcmp(stype, "shiny") == 0){
        Phong *c = new Phong(line);
        Phong *dev_c = 0;
        cudaMalloc((void**)&dev_c, sizeof(Phong));
        cudaMemcpy(dev_c, c, sizeof(Phong), cudaMemcpyHostToDevice);
        shaders.push_back(dev_c);
        delete c;
      }
    }
    else if (strcmp(object, "camera:") == 0){
      Camera *cam = new Camera (line);
      cam->beginFrame();
      cudaMalloc((void**)&camera, sizeof(Camera));
      cudaMemcpy(camera, cam, sizeof(Camera), cudaMemcpyHostToDevice);
      delete cam;
    }
    else if (strcmp(object, "light:") == 0){
      Light *_l = new Light (line);
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
