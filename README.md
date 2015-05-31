# CUDA Raytracer

## System Configuration
- GPU: NVIDIA GeForce 610M
- compiled with NVCC 6.0.1
- nvidia driver v331.133
- Ubuntu 14.04 LTS
- running bumblebee 3.2.1

rdc means 'relocatable device code'' (enable to enable inheritance and virtual functions)

Вefore running optirun (required to run on GPU because of Bumblebee) run
```
update-alternatives --config x86_64-linux-gnu_gl_conf
```
and select /usr/lib/nvidia-331-updates-prime/ld.so.conf

TODO:
- add functions to allocate memory in both places (wrap new, cudaMalloc, cudaMemcpy, free)
- clean up unnecessary (?) functions, if any
- 

