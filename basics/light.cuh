#ifndef __LIGHT_H__
#define __LIGHT_H__

#include <utils/color.cuh>

struct Light {
  Vector pos;
  Color color;
  float power;

  inline Light() {}
  inline Light(const char* str){
    sscanf(str, "%f%f%f%lf%lf%lf%f",
                &color.r, &color.g, &color.b,
                &pos.x, &pos.y, &pos.z, &power);
  }
};

#endif
