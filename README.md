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
 raytracer threadCount numBuckets fileName
```

*threadCount* - must be present, the number of threads on which the raytracer will run
*numBlocks* - optional, the number of buckets in a row. Every thread is responsible for rendering a single bucket, and naturally, *numBuckets* should divide *threadCount*
*fileName* - the name for a file where the resulting picture should be stored. Currently supports only bitmap files
