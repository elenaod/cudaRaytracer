#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <basics/ray.cuh>

// hm... lightPos, lightPower, lightColor should come here?!
class Camera {
  Vector upLeft, upRight, downLeft;
public:
  Vector pos; // position
  double yaw, pitch, roll; // in degrees
  double fov; // in degrees
  double aspect; // 1.3 or ?

  void beginFrame(void);
  __device__
  Ray getScreenRay(double x, double y) const;
};

#endif // __CAMERA_H__
