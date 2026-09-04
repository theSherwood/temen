/* Varvara — the Uxn device layer (https://wiki.xxiivv.com/site/varvara.html), the subset a
 * framebuffer-and-keyboard host can serve: System (palette, expansion, halt), Console (output),
 * Screen (two 2-bit layers, pixel/fill/sprite ops, composed to RGBA), Controller (buttons + key),
 * Datetime (a deterministic virtual clock). Mouse, Audio and File are absent: the playground has no
 * such capabilities yet, so writes to those pages are inert and reads return the device byte.
 *
 * Freestanding, and shared verbatim by the reactor guest (main.c) and the headless differential
 * (uxn_diff.c): the compositor is a pure function of the ROM + the key script, so the frames a native
 * `cc` build and the sandboxed guest present are byte-identical. The platform supplies
 * `varvara_console_write` (stdout) and `malloc`. */
#include "uxn.h"

extern void *malloc(unsigned long n);
extern void free(void *p);
void varvara_console_write(const Uint8 *buf, int len);

#define SCREEN_MAX 1024 /* per axis */

typedef unsigned int Uint32;

typedef struct {
  int width, height;
  Uint8 *bg, *fg;   /* width*height 2-bit pixels (values 0..3), one byte each */
  Uint8 *rgba;      /* the composed frame, width*height*4 */
  int dirty;        /* a draw or palette change since the last compose */
  Uint32 palette[4];
} Screen;

static Screen scr;
static unsigned long ticks; /* frames since reset — the Datetime device's clock */

/* ---- System ------------------------------------------------------------------------------------ */

static void system_palette(Uxn *u) {
  Uint16 r = PEEK2(u->dev + 0x08), g = PEEK2(u->dev + 0x0a), b = PEEK2(u->dev + 0x0c);
  for (int i = 0; i < 4; i++) {
    int sh = 12 - i * 4;
    Uint32 cr = (r >> sh) & 0xf, cg = (g >> sh) & 0xf, cb = (b >> sh) & 0xf;
    scr.palette[i] = (cr * 0x11) | (cg * 0x11) << 8 | (cb * 0x11) << 16 | 0xff000000u;
  }
  scr.dirty = 1;
}

/* The expansion port: a command struct in memory. 0 = fill, 1 = cpyl (ascending copy), 2 = cpyr. */
static void system_expansion(Uxn *u) {
  Uint8 *cmd = u->ram + PEEK2(u->dev + 0x02);
  Uint16 len = PEEK2(cmd + 1);
  switch (cmd[0]) {
  case 0: {
    Uint8 *dst = u->ram + (PEEK2(cmd + 3) % UXN_BANKS) * 0x10000;
    Uint16 a = PEEK2(cmd + 5);
    for (Uint16 i = 0; i < len; i++) dst[(Uint16)(a + i)] = cmd[7];
    break;
  }
  case 1: case 2: {
    Uint8 *src = u->ram + (PEEK2(cmd + 3) % UXN_BANKS) * 0x10000;
    Uint8 *dst = u->ram + (PEEK2(cmd + 7) % UXN_BANKS) * 0x10000;
    Uint16 sa = PEEK2(cmd + 5), da = PEEK2(cmd + 9);
    if (cmd[0] == 1)
      for (Uint16 i = 0; i < len; i++) dst[(Uint16)(da + i)] = src[(Uint16)(sa + i)];
    else
      for (Uint16 i = len; i-- > 0;) dst[(Uint16)(da + i)] = src[(Uint16)(sa + i)];
    break;
  }
  default: break;
  }
}

static void system_debug(Uxn *u) {
  Uint8 line[3 * 8 + 6];
  static const char hex[] = "0123456789abcdef";
  for (int st = 0; st < 2; st++) {
    Stack *s = st ? &u->rst : &u->wst;
    int n = 0;
    line[n++] = st ? 'R' : 'W'; line[n++] = 'S'; line[n++] = 'T'; line[n++] = ' ';
    for (int i = 8; i > 0; i--) {
      Uint8 v = s->dat[(Uint8)(s->ptr - i)];
      line[n++] = hex[v >> 4]; line[n++] = hex[v & 0xf]; line[n++] = i == 1 ? '<' : ' ';
    }
    line[n++] = '\n';
    varvara_console_write(line, n);
  }
}

/* ---- Screen ------------------------------------------------------------------------------------ */

static void screen_resize(int w, int h) {
  if (w < 1) w = 1;
  if (h < 1) h = 1;
  if (w > SCREEN_MAX) w = SCREEN_MAX;
  if (h > SCREEN_MAX) h = SCREEN_MAX;
  if (w == scr.width && h == scr.height && scr.bg) return;
  free(scr.bg); free(scr.fg); free(scr.rgba);
  scr.width = w; scr.height = h;
  scr.bg = malloc((unsigned long)(w * h));
  scr.fg = malloc((unsigned long)(w * h));
  scr.rgba = malloc((unsigned long)(w * h * 4));
  for (int i = 0; i < w * h; i++) scr.bg[i] = scr.fg[i] = 0;
  scr.dirty = 1;
}

static void screen_fill(Uint8 *layer, int x1, int y1, int x2, int y2, Uint8 color) {
  if (x2 > scr.width) x2 = scr.width;
  if (y2 > scr.height) y2 = scr.height;
  for (int y = y1; y < y2; y++)
    for (int x = x1; x < x2; x++) layer[y * scr.width + x] = color;
  scr.dirty = 1;
}

/* Blending: row = the sprite pixel's 2-bit value, column = the op's color nibble; row 4 says whether
 * a 0 pixel is drawn (opaque) or left alone (transparent). */
static const Uint8 blending[5][16] = {
  {0, 0, 0, 0, 1, 0, 1, 1, 2, 2, 0, 2, 3, 3, 3, 0},
  {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
  {1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1},
  {2, 3, 1, 2, 2, 3, 1, 2, 2, 3, 1, 2, 2, 3, 1, 2},
  {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};

static void screen_blit(Uint8 *layer, const Uint8 *sprite, int x, int y, Uint8 color, int flipx,
                        int flipy, int twobpp) {
  int opaque = blending[4][color];
  for (int v = 0; v < 8; v++) {
    Uint8 c1 = sprite[v], c2 = twobpp ? sprite[v + 8] : 0;
    for (int h = 7; h >= 0; h--, c1 >>= 1, c2 >>= 1) {
      Uint8 ch = (c1 & 1) | ((c2 << 1) & 2);
      if (opaque || ch) {
        int xx = x + (flipx ? 7 - h : h), yy = y + (flipy ? 7 - v : v);
        if (xx >= 0 && xx < scr.width && yy >= 0 && yy < scr.height)
          layer[yy * scr.width + xx] = blending[ch][color];
      }
    }
  }
  scr.dirty = 1;
}

static void screen_deo(Uxn *u, Uint8 port) {
  Uint8 *d = u->dev + 0x20;
  switch (port) {
  case 0x3: screen_resize(PEEK2(d + 2), scr.height); break;
  case 0x5: screen_resize(scr.width, PEEK2(d + 4)); break;
  case 0xe: { /* pixel */
    Uint8 ctrl = d[0xe], color = ctrl & 3;
    Uint8 *layer = ctrl & 0x40 ? scr.fg : scr.bg;
    int x = PEEK2(d + 8), y = PEEK2(d + 0xa);
    if (ctrl & 0x80) { /* fill from (x,y) to the edge the flip bits pick */
      int x2 = scr.width, y2 = scr.height;
      if (ctrl & 0x10) { x2 = x; x = 0; }
      if (ctrl & 0x20) { y2 = y; y = 0; }
      screen_fill(layer, x, y, x2, y2, color);
    } else {
      if (x < scr.width && y < scr.height) layer[y * scr.width + x] = color;
      if (d[0x6] & 0x1) POKE2(d + 8, x + 1);
      if (d[0x6] & 0x2) POKE2(d + 0xa, y + 1);
      scr.dirty = 1;
    }
    break;
  }
  case 0xf: { /* sprite */
    Uint8 ctrl = d[0xf], move = d[0x6], length = move >> 4, twobpp = !!(ctrl & 0x80);
    Uint8 *layer = ctrl & 0x40 ? scr.fg : scr.bg;
    Uint16 x = PEEK2(d + 8), y = PEEK2(d + 0xa), addr = PEEK2(d + 0xc);
    int dx = (move & 0x1) << 3, dy = (move & 0x2) << 2;
    int flipx = ctrl & 0x10, flipy = ctrl & 0x20, fx = flipx ? -1 : 1, fy = flipy ? -1 : 1;
    for (int i = 0; i <= length; i++) {
      /* a run of sprites lays out along the axis the auto flag does NOT advance */
      screen_blit(layer, u->ram + addr, (Sint16)(x + dy * fx * i), (Sint16)(y + dx * fy * i),
                  ctrl & 0xf, flipx, flipy, twobpp);
      addr = (Uint16)(addr + ((move & 0x04) << (1 + twobpp)));
    }
    POKE2(d + 8, x + dx * fx);
    POKE2(d + 0xa, y + dy * fy);
    POKE2(d + 0xc, addr);
    break;
  }
  default: break;
  }
}

/* ---- Datetime — a virtual clock: 60 ticks per second from 2026-01-01 00:00:00 (a Thursday) ------ */

static Uint8 datetime_dei(Uint8 port) {
  static const Uint8 mdays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  unsigned long secs = ticks / 60;
  int day = (int)(secs / 86400) % 365, tod = (int)(secs % 86400);
  int month = 0, doty = day;
  while (month < 11 && day >= mdays[month]) day -= mdays[month++];
  switch (port) {
  case 0x0: return 2026 >> 8;
  case 0x1: return 2026 & 0xff;
  case 0x2: return (Uint8)month;
  case 0x3: return (Uint8)(day + 1);
  case 0x4: return (Uint8)(tod / 3600);
  case 0x5: return (Uint8)(tod / 60 % 60);
  case 0x6: return (Uint8)(tod % 60);
  case 0x7: return (Uint8)((4 + doty) % 7);
  case 0x8: return (Uint8)(doty >> 8);
  case 0x9: return (Uint8)doty;
  default: return 0;
  }
}

/* ---- The device hooks the CPU calls ----------------------------------------------------------- */

Uint8 uxn_dei(Uxn *u, Uint8 port) {
  switch (port) {
  case 0x04: return u->wst.ptr;
  case 0x05: return u->rst.ptr;
  case 0x22: return (Uint8)(scr.width >> 8);
  case 0x23: return (Uint8)scr.width;
  case 0x24: return (Uint8)(scr.height >> 8);
  case 0x25: return (Uint8)scr.height;
  default:
    if ((port & 0xf0) == 0xc0) return datetime_dei(port & 0xf);
    return u->dev[port];
  }
}

void uxn_deo(Uxn *u, Uint8 port) {
  switch (port & 0xf0) {
  case 0x00:
    if (port == 0x03) system_expansion(u);
    else if (port >= 0x08 && port <= 0x0d) system_palette(u);
    else if (port == 0x0e) system_debug(u);
    break;
  case 0x10:
    if (port == 0x18 || port == 0x19) varvara_console_write(u->dev + port, 1);
    break;
  case 0x20: screen_deo(u, port & 0xf); break;
  default: break;
  }
}

/* ---- The emulator-facing API ------------------------------------------------------------------ */

void varvara_init(Uxn *u) {
  for (int i = 0; i < 256; i++) u->dev[i] = 0;
  u->wst.ptr = u->rst.ptr = 0;
  ticks = 0;
  screen_resize(512, 320); /* the spec's default; a ROM sets its own through Screen/width,height */
  POKE2(u->dev + 0x22, scr.width);
  POKE2(u->dev + 0x24, scr.height);
  system_palette(u);
}

int varvara_halted(Uxn *u) { return u->dev[0x0f] != 0; }

/* Controller: a button-mask change (`button`, bits A B Select Start Up Down Left Right) and/or a key
 * byte, each firing the vector; the key byte is cleared afterwards like the reference does. */
void varvara_controller(Uxn *u, Uint8 button, Uint8 key) {
  Uint8 *d = u->dev + 0x80;
  d[2] = button;
  d[3] = key;
  uxn_eval(u, PEEK2(d));
  d[3] = 0;
}

/* One frame: fire the screen vector, advance the clock. */
void varvara_screen_vector(Uxn *u) {
  uxn_eval(u, PEEK2(u->dev + 0x20));
  ticks++;
}

/* Compose the layers to RGBA if anything changed. Returns the frame (NULL when nothing changed and
 * `force` is 0) — `*w`/`*h` receive its size. */
const Uint8 *varvara_frame(int force, int *w, int *h) {
  *w = scr.width;
  *h = scr.height;
  if (!scr.dirty && !force) return 0;
  Uint32 *out = (Uint32 *)scr.rgba;
  int n = scr.width * scr.height;
  for (int i = 0; i < n; i++) {
    Uint8 f = scr.fg[i];
    out[i] = scr.palette[f ? f : scr.bg[i]];
  }
  scr.dirty = 0;
  return scr.rgba;
}
