SHELL = /bin/bash

OBJS = matrix.cu sdl.cu \
       camera.cu geometry.cu shading.cu \
       init.cu main.cu

FLAGS = -I/usr/include/SDL -D_GNU_SOURCE=1 -D_REENTRANT -L/usr/lib/x86_64-linux-gnu -lSDL
NVCC = /usr/local/cuda-6.0/bin/nvcc
COMPILE = $(NVCC) $(FLAGS) 

raytracer: $(OBJS)
	$(COMPILE) --gpu-architecture=sm_20 -rdc=true $(OBJS) -o raytracer

clean:
	rm raytracer *~
