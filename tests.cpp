#include <cstdio>

void init(int m[3][3]){
  m[0][0] = m[1][1] = m[2][2] = 5;
  m[0][1] = m[0][2] = 1;
  m[1][0] = m[1][2] = 2;
  m[2][0] = m[2][1] = 3;
}

void print(int m[3][3]){
  for(int i = 0; i < 3; ++i){
    for(int j = 0; j < 3; ++j){
      printf("%d ",m[i][j]);
    }
    printf("\n");
  }
}

int main(){
  int m[3][3];
  init(m);
  print(m);
  return 0;
}
