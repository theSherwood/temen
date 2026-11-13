/* surface.c — a WebGPU surface follows its embedder's size (#2837). A guest draws a frame at each
 * size the embedder gives the surface, asking for a new size through the test's `resize` cap between
 * frames, as a window resizes under a running program:
 *   - each frame is cleared to its own colour, copied out and read back, and every pixel compared;
 *   - the `surface` op reports each new size before the frame is drawn at it;
 *   - a size with a zero side gives nothing to draw: the stream still runs, the `error` op says why,
 *     and the host shows nothing.
 * The same program runs on a window's surface and on an offscreen one. Each colour is a whole step
 * of 1/255 and a clear covers every pixel, so any conformant WebGPU gives exactly these bytes. */

#include <stdio.h>

#ifdef TEMEN_GUEST
#include "temen_gpu.h"
#endif

#define MAX_W 96
#define MAX_H 80

static gpu_stream s;
static unsigned char stream[4 << 10];
/* A row of the readback is padded to a multiple of 256 bytes: 512 at the widest frame. */
static unsigned char px[512 * MAX_H];
static char why[256];

enum { FRAME = 1, FRAME_VIEW, READBACK };

/* Asks the embedder to resize the surface: the test's own cap, not the protocol's. */
static void resize(long w, long h) {
  static long cap = -2;
  if (cap == -2) cap = __vm_cap_resolve("resize", 6);
  __vm_host_call(cap, 0, w, h, 0, 0);
}

static int contains(const char *a, const char *b) {
  for (; *a; a++) {
    const char *p = a, *q = b;
    while (*p && *q && *p == *q) p++, q++;
    if (!*q) return 1;
  }
  return 0;
}

/* Draws one frame at the size the surface reports, which must be `w` by `h`, cleared to `rgba`,
 * and checks every pixel it reads back. 1 if all is as it should be. */
static int frame(unsigned w, unsigned h, const unsigned char rgba[4]) {
  gpu_surface_info info;
  if (gpu_surface(&info) != 0 || info.width != w || info.height != h) {
    printf("frame %ux%u: the surface reports %ux%u\n", w, h, (unsigned)info.width,
           (unsigned)info.height);
    return 0;
  }
  int bgra = info.format == GPU_BGRA8UNORM;
  unsigned row = (w * 4 + 255) / 256 * 256;
  gpu_create_buffer(&s, READBACK, row * h, GPU_BUFFER_COPY_DST | GPU_BUFFER_COPY_SRC);
  gpu_acquire(&s, FRAME, FRAME_VIEW);
  gpu_color_attachment color = {FRAME_VIEW, 0, GPU_LOAD_CLEAR, GPU_STORE_STORE,
                                {rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0,
                                 rgba[3] / 255.0}};
  gpu_begin_render_pass(&s, 1, &color);
  gpu_end_pass(&s);
  gpu_copy_texture_to_buffer(&s, FRAME, 0, 0, w, h, READBACK, 0, row);
  gpu_readback(&s, READBACK, 0, row * h);
  gpu_present(&s, FRAME, FRAME_VIEW);
  long r = gpu_submit(&s);
  if (r != 0) {
    gpu_error(why, sizeof why);
    printf("frame %ux%u: submit %ld (%s)\n", w, h, r, why);
    return 0;
  }
  long n = gpu_read_wait(READBACK, px, sizeof px);
  gpu_destroy(&s, READBACK);
  gpu_submit(&s);
  if (n != (long)(row * h)) {
    gpu_error(why, sizeof why);
    printf("frame %ux%u: read %ld bytes (%s)\n", w, h, n, why);
    return 0;
  }
  int bad = 0;
  for (unsigned y = 0; y < h; y++)
    for (unsigned x = 0; x < w; x++) {
      const unsigned char *p = &px[y * row + x * 4];
      unsigned char got[4] = {p[bgra ? 2 : 0], p[1], p[bgra ? 0 : 2], p[3]};
      bad += got[0] != rgba[0] || got[1] != rgba[1] || got[2] != rgba[2] || got[3] != rgba[3];
    }
  printf("frame %ux%u: %s, %d pixels differ\n", w, h, bad ? "FAIL" : "OK", bad);
  return bad == 0;
}

int main(void) {
  gpu_stream_init(&s, stream, sizeof stream);
  static const unsigned char red[4] = {255, 0, 0, 255}, green[4] = {0, 255, 0, 255},
                             blue[4] = {0, 0, 255, 255};
  int ok = frame(64, 48, red);

  resize(96, 80);
  ok &= frame(96, 80, green);

  /* No area: the stream runs into a stand-in, and the error op says why. */
  resize(0, 0);
  gpu_surface_info info;
  int empty = gpu_surface(&info) == 0 && info.width == 0 && info.height == 0;
  gpu_acquire(&s, FRAME, FRAME_VIEW);
  gpu_present(&s, FRAME, FRAME_VIEW);
  long r = gpu_submit(&s);
  gpu_error(why, sizeof why);
  int refused = empty && r == 0 && contains(why, "no area");
  printf("no area: %s (%s)\n", refused ? "OK" : "FAIL", why);
  ok &= refused;

  resize(40, 30);
  ok &= frame(40, 30, blue);

  printf("surface: %s\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}
