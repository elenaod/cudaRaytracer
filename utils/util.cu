#include <utils/util.cuh>

bool getLineFrom(int f, char* line){
  char c; int i = 0;
  while( read(f, &c, 1) > 0 ){
    if (c == '\n'){ line[i] = '\0'; return true;}
    line[i++] = c;
  }
  line[i] = '\0';
  return false;
}

bool getCommandLineArgs(int argc, char** argv,
                        unsigned& threadCount, unsigned& numBuckets,
                        unsigned& resX, unsigned& resY,
                        char* inputFile, char* outputFile) {
  unsigned i = 1;
  bool fail = true;

  for(; i < argc;){
    if (strcmp(argv[i], "--threadCount") == 0){
      threadCount = atoi(argv[i + 1]); i += 1; fail = false;
    }
    else if (strcmp(argv[i], "--inputFile") == 0){
      strcpy(inputFile, argv[i + 1]); i += 2; fail = false;
    }
    else if (strcmp(argv[i], "--numBuckets") == 0){
      numBuckets = atoi(argv[i + 1]); i += 2;
    }
    else if (strcmp(argv[i], "--outputFile") == 0){
      strcpy(outputFile, argv[i + 1]); i += 2;
    }
    else if (strcmp(argv[i], "--resX") == 0){
      resX = atoi(argv[i + 1]); i += 2;
    }
    else if (strcmp(argv[i], "--resY") == 0){
      resY = atoi(argv[i + 1]); i += 2;
    }
    else {
      printf("Unknown argument!"); fail = true;
      return fail;
    }
  }

  return fail;
}

cudaError_t moveToDevice(void* src, void** dest, size_t bytes,
                         bool freeOnHost) {
  cudaMalloc(dest, bytes);
  cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice);
  switch (freeOnHost) {
    case true: {free(src); break;}
    case false: {break;}
  }
  return cudaGetLastError();
}
