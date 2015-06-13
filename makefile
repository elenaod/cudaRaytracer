SHELL = /bin/bash

PROJECT_DIR = /home/saffi/dev/parallel/cudaRaytracer

OBJS = utils/matrix.cu utils/sdl.cu utils/util.cu \
       basics/camera.cu \
       geometries/plane.cu geometries/sphere.cu \
       shaders/Phong.cu shaders/CheckerShader.cu \
       main/scene.cu main/kernels.cu

MAIN = main/main.cu
TIMING = main/timing_main.cu
DISCO_PLANE = main/disco.cu

FLAGS = -I/usr/include/SDL -I$(PROJECT_DIR) \
        -D_GNU_SOURCE=1 -D_REENTRANT \
        -L/usr/lib/x86_64-linux-gnu -lSDL \
        --gpu-architecture=sm_20 -rdc=true

NVCC = /usr/local/cuda-6.0/bin/nvcc
COMPILE = $(NVCC) $(FLAGS) 

#timing: $(OBJS) $(TIMING)
#	$(COMPILE) $(OBJS) $(TIMING) -o timing

disco: $(OBJS) $(DISCO_PLANE)
	$(COMPILE) $(OBJS) $(DISCO_PLANE) -o disco

raytracer: $(OBJS) $(MAIN)
	$(COMPILE) $(OBJS) $(MAIN) -o raytracer

clean:
	rm raytracer *~
