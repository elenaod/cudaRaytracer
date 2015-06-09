#include <SDL/SDL.h>
#include <stdio.h>
#include "sdl.cuh"

bool initGraphics(SDL_Surface** _screen,
                 int frameWidth, int frameHeight) {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    printf("Cannot initialize SDL: %s\n", SDL_GetError());
    return false;
  }
  *(_screen) = SDL_SetVideoMode(frameWidth, frameHeight, 32, 0);
  if (!*(_screen)) {
    printf("Cannot set video mode %dx%d - %s\n",
           frameWidth, frameHeight, SDL_GetError());
    return false;
  }
  return true;
}

void closeGraphics(void)
{
  SDL_Quit();
}

/// displays a VFB (virtual frame buffer) to the real framebuffer, with the necessary color clipping
void displayVFB(SDL_Surface* _screen, Color* vfb)
{
  int rs = _screen->format->Rshift;
  int gs = _screen->format->Gshift;
  int bs = _screen->format->Bshift;
  for (int y = 0; y < _screen->h; y++) {
    Uint32 *row = (Uint32*) ((Uint8*)
                           _screen->pixels + y * _screen->pitch);
    for (int x = 0; x < _screen->w; x++)
      row[x] = vfb[y * VFB_MAX_SIZE + x].toRGB32(rs, gs, bs);
  }
  SDL_Flip(_screen);
}

/// waits the user to indicate he wants to close the application (by either clicking on the "X" of the window,
/// or by pressing ESC)
void waitForUserExit(void)
{
  SDL_Event ev;
  while (1) {
    while (SDL_WaitEvent(&ev)) {
      switch (ev.type) {
        case SDL_QUIT:
          return;
        case SDL_KEYDOWN:
        {
          switch (ev.key.keysym.sym) {
            case SDLK_ESCAPE:
              return;
            default:
              break;
          }
        }
        default:
          break;
      }
    }
  }
}

int frameWidth(SDL_Surface* screen)
{
  if (screen) return screen->w;
  return 0;
}

int frameHeight(SDL_Surface* screen)
{
  if (screen) return screen->h;
  return 0;
}
