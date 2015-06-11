#ifndef __SDL_H__
#define __SDL_H__

#include "color.cuh"
#include "constants.cuh"
#include <SDL/SDL.h>

bool initGraphics(SDL_Surface** _screen,
                  int frameWidth, int frameHeight);
void closeGraphics(void);

void displayVFB(SDL_Surface* _screen, Color* vfb);

void waitForUserExit(void);

int frameWidth(void);
int frameHeight(void);

#endif // __SDL_H__

