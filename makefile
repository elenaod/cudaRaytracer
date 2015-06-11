SHELL = /bin/bash

PROJECT_DIR = /home/saffi/dev/parallel/cudaRaytracer

OBJS = utils/matrix.cu utils/sdl.cu \
       basics/camera.cu \
       geometries/plane.cu geometries/sphere.cu \
       shaders/Phong.cu shaders/CheckerShader.cu \
       init.cu kernels.cu main.cu

FLAGS = -I/usr/include/SDL -I$(PROJECT_DIR) \
        -D_GNU_SOURCE=1 -D_REENTRANT \
        -L/usr/lib/x86_64-linux-gnu -lSDL \
        --gpu-architecture=sm_20 -rdc=true

NVCC = /usr/local/cuda-6.0/bin/nvcc
COMPILE = $(NVCC) $(FLAGS) 

raytracer: $(OBJS)
	$(COMPILE) $(OBJS) -o raytracer

clean:
	rm raytracer *~
