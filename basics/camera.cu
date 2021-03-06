#include <basics/camera.cuh>
#include <utils/matrix.cuh>
#include <utils/util.cuh>
#include <utils/sdl.cuh>

Camera::Camera() {}
Camera::Camera(const char* str){
  float aspectX, aspectY;
  sscanf(str, "%lf%lf%lf%lf%lf%lf%lf%f%f",
               &yaw, &pitch, &roll,
               &pos.x, &pos.y, &pos.z,
               &fov, &aspectX, &aspectY);
  aspect = aspectX / aspectY;
}

void Camera::beginFrame(void) {
  double x = -aspect;
  double y = +1;

  Vector corner = Vector(x, y, 1);
  Vector center = Vector(0, 0, 1);

  double lenXY = (corner - center).length();
  double wantedLength = tan(toRadians(fov / 2));

  double scaling = wantedLength / lenXY;

  x *= scaling;
  y *= scaling;

  this->upLeft = Vector(x, y, 1);
  this->upRight = Vector(-x, y, 1);
  this->downLeft = Vector(x, -y, 1);

  Matrix rotation = rotZ(toRadians(roll))
                  * rotX(toRadians(pitch))
                  * rotY(toRadians(yaw));
  upLeft *= rotation;
  upRight *= rotation;
  downLeft *= rotation;

  upLeft += pos;
  upRight += pos;
  downLeft += pos;
}

__device__
Ray Camera::getScreenRay(double x, double y) const{
  Ray result; // A, B -     C = A + (B - A) * x
  result.start = this->pos;
  Vector target = upLeft + 
    (upRight - upLeft) * (x / (double) RESX) +
    (downLeft - upLeft) * (y / (double) RESY);
  // A - camera; B = target
  result.dir = target - this->pos;
  
  result.dir.normalize();
  return result;
}
