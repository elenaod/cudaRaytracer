# CUDA Raytracer

A CUDA Raytracer based heavily on the Trinity raytracer project (https://github.com/anrieff/trinity).
Currently supports
- Plane and Spehere geometries
- Checker Shader
- antialiasing

## System Configuration
- GPU: NVIDIA GeForce 610M
- compiled with NVCC 6.0.1
- nvidia driver v331.133
- Ubuntu 14.04 LTS
- running bumblebee 3.2.1

Вefore running optirun (required to run on GPU because of Bumblebee) run
```
update-alternatives --config x86_64-linux-gnu_gl_conf
```
and select /usr/lib/nvidia-331-updates-prime/ld.so.conf

## Arguments
To run, use
```
 raytracer
```
The command line options are
  - *--threadCount*  The number of threads on which the algorithm runs
  - *--inputFile*    The file from which a scene is read
  - *--outputFile*   The file to which the result image should be saved, optional
  - *--resX*         The width of the image, optional, default = 640
  - *--resY*         The height of the image, optional, default = 480
  - *--numBuckets*   The number of horizontal boxes in which the image is partioned for rendering

