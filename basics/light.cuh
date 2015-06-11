#ifndef __LIGHT_H__
#define __LIGHT_H__

#include <utils/color.cuh>

struct Light {
  Vector pos;
  Color color;
  float power;
};

#endif
