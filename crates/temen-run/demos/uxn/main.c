/* Uxn in the playground — the reactor guest (the bounce.c / Doom shape). `_start → main` resolves the
 * `display`, `keyboard` and `fs` capabilities, reads the ROM served as "boot.rom" through `fs`, and runs
 * its reset vector. The page then calls `tick()` once per animation frame: drain the key events into the
 * Controller device, fire the Screen vector, and present the composed frame through `display` when it
 * changed. When the ROM halts (System/state), the guest exits, which ends the reactor loop. */
#include "uxn.c"
#include "varvara.c"

extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);
extern long write(int fd, const void *buf, unsigned long n);
extern void exit(int code);

#define ROM_MAX 0xff00

static Uxn u;
static int disp, kbd, fs;

void varvara_console_write(const Uint8 *buf, int len) { write(1, buf, (unsigned long)len); }

/* JS keyCode → the Controller's button bit, or 0 for a non-button key. */
static Uint8 button_of(int code) {
  switch (code) {
  case 17: return 0x01; /* Ctrl  = A      */
  case 18: return 0x02; /* Alt   = B      */
  case 16: return 0x04; /* Shift = Select */
  case 36: return 0x08; /* Home  = Start  */
  case 38: return 0x10; /* Up    */
  case 40: return 0x20; /* Down  */
  case 37: return 0x40; /* Left  */
  case 39: return 0x80; /* Right */
  default: return 0;
  }
}

/* JS keyCode → the ASCII byte a key-down delivers, or 0. */
static Uint8 key_of(int code, Uint8 buttons) {
  if (code >= 65 && code <= 90) return (Uint8)(buttons & 0x04 ? code : code + 32);
  if (code >= 48 && code <= 57) return (Uint8)code;
  switch (code) {
  case 32: return ' ';
  case 13: return 0x0d;
  case 8: return 0x08;
  case 9: return 0x09;
  case 27: return 0x1b;
  default: return 0;
  }
}

int main(void) {
  disp = __vm_cap_resolve("display", 7);
  kbd = __vm_cap_resolve("keyboard", 8);
  fs = __vm_cap_resolve("fs", 2);
  u.ram = malloc(UXN_BANKS * 0x10000);
  for (long i = 0; i < UXN_BANKS * 0x10000; i++) u.ram[i] = 0;
  varvara_init(&u);
  if (fs >= 0) {
    static const char name[] = "boot.rom";
    long fd = __vm_host_call(fs, 0, (long)name, 8, 0, 0);
    if (fd >= 0) {
      long got = 0, n;
      while (got < ROM_MAX && (n = __vm_host_call(fs, 1, fd, (long)(u.ram + 0x100 + got), ROM_MAX - got, 0)) > 0)
        got += n;
      __vm_host_call(fs, 4, fd, 0, 0, 0);
    }
  }
  uxn_eval(&u, 0x100);
  return 0;
}

int tick(void) {
  static Uint8 buttons;
  for (;;) {
    long e = __vm_host_call(kbd, 0, 0, 0, 0, 0);
    if (e < 0) break;
    int code = (int)(e & 0xffff), pressed = (int)((e >> 16) & 1);
    Uint8 bit = button_of(code);
    if (bit) {
      Uint8 next = pressed ? (Uint8)(buttons | bit) : (Uint8)(buttons & ~bit);
      if (next == buttons) continue;
      buttons = next;
      varvara_controller(&u, buttons, 0);
    } else if (pressed) {
      Uint8 key = key_of(code, buttons);
      if (key) varvara_controller(&u, buttons, key);
    }
  }
  varvara_screen_vector(&u);
  int w, h;
  const Uint8 *frame = varvara_frame(0, &w, &h);
  if (frame && disp >= 0) __vm_host_call(disp, 0, (long)frame, w, h, 0);
  if (varvara_halted(&u)) exit(0);
  return 0;
}
