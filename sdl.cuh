#ifndef __SDL_H__
#define __SDL_H__

#include "color.cuh"
#include "constants.cuh"
#include <SDL/SDL.h>

bool initGraphics(SDL_Surface** _screen,
                  int frameWidth, int frameHeight);
void closeGraphics(void);

//!< displays the VFB (Virtual framebuffer) to the real one.
void displayVFB(SDL_Surface* _screen,
                Color** vfb);

//!< Pause. Wait until the user closes the application
void waitForUserExit(void);

/*int frameWidth(void); //!< returns the frame width (pixels)
int frameHeight(void); //!< returns the frame height (pixels) */

#endif // __SDL_H__

