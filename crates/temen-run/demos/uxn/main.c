/* Uxn in the playground — the reactor guest (the bounce.c / Doom shape). `_start → main` resolves the
 * `display`, `keyboard`, `mouse` and `fs` capabilities, loads the program served through `fs` — Uxntal
 * source as "boot.tal" (assembled here, in the sandbox, by uxnasm_core.c) or a ROM as "boot.rom" — and
 * runs its reset vector. The page then calls `tick()` once per animation frame: drain the key
 * events into the Controller device and the pointer events into the Mouse device, fire the Screen
 * vector, and present the composed frame through `display` when it changed. When the ROM halts
 * (System/state), the guest exits, which ends the reactor loop.
 *
 *   k = __vm_cap_resolve("keyboard", 8);  e = __vm_host_call(k, 0, 0,0,0,0);  // (pressed<<16)|keyCode | -1
 *   m = __vm_cap_resolve("mouse", 5);     e = __vm_host_call(m, 0, 0,0,0,0);  // (kind<<32)|payload | -1
 * mouse kind 0 = pointer: payload (buttons<<24)|(x<<12)|y; kind 1 = wheel: (dx&0xffff)<<16|(dy&0xffff). */
#include "uxn.c"
#include "varvara.c"
#include "uxnasm_core.c"

extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);
extern long write(int fd, const void *buf, unsigned long n);
extern void exit(int code);

#define ROM_MAX 0xff00
#define TAL_MAX (1 << 20)

static Uxn u;
static int disp, kbd, mouse, fs;
static int asm_failed; /* boot.tal did not assemble: the first tick reports it and exits */

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

/* JS keyCode → the ASCII byte a key-down delivers, or 0. Shift (the Select button) picks the upper
 * case / the shifted punctuation, the US layout. */
static Uint8 key_of(int code, Uint8 buttons) {
  int shift = buttons & 0x04;
  if (code >= 65 && code <= 90) return (Uint8)(shift ? code : code + 32);
  if (code >= 48 && code <= 57) return (Uint8)(shift ? ")!@#$%^&*("[code - 48] : code);
  switch (code) {
  case 186: return shift ? ':' : ';';
  case 187: return shift ? '+' : '=';
  case 188: return shift ? '<' : ',';
  case 189: return shift ? '_' : '-';
  case 190: return shift ? '>' : '.';
  case 191: return shift ? '?' : '/';
  case 192: return shift ? '~' : '`';
  case 219: return shift ? '{' : '[';
  case 220: return shift ? '|' : '\\';
  case 221: return shift ? '}' : ']';
  case 222: return shift ? '"' : '\'';
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
  mouse = __vm_cap_resolve("mouse", 5);
  fs = __vm_cap_resolve("fs", 2);
  u.ram = malloc(UXN_BANKS * 0x10000);
  for (long i = 0; i < UXN_BANKS * 0x10000; i++) u.ram[i] = 0;
  varvara_init(&u);
  if (fs >= 0) {
    static const char tal[] = "boot.tal", rom[] = "boot.rom";
    long fd = __vm_host_call(fs, 0, (long)tal, 8, 0, 0);
    if (fd >= 0) { /* Uxntal source: read it all, assemble straight into bank 0 */
      char *src = malloc(TAL_MAX);
      long got = 0, n;
      while (got < TAL_MAX && (n = __vm_host_call(fs, 1, fd, (long)(src + got), TAL_MAX - got, 0)) > 0)
        got += n;
      __vm_host_call(fs, 4, fd, 0, 0, 0);
      int rom_len;
      asm_failed = !uxnasm_assemble(src, (int)got, u.ram, &rom_len);
      free(src);
    } else if ((fd = __vm_host_call(fs, 0, (long)rom, 8, 0, 0)) >= 0) {
      long got = 0, n;
      while (got < ROM_MAX && (n = __vm_host_call(fs, 1, fd, (long)(u.ram + 0x100 + got), ROM_MAX - got, 0)) > 0)
        got += n;
      __vm_host_call(fs, 4, fd, 0, 0, 0);
    }
  }
  if (!asm_failed) uxn_eval(&u, 0x100);
  return 0;
}

/* "uxnasm: line N: <message>\n" on stdout — the page shows a reactor's stdout, so the editor's author
 * sees where the source broke. */
static void report_asm_error(void) {
  char buf[ASM_TOKEN + 96];
  int n = 0, line = uxnasm_error_line, digits = 1;
  for (const char *p = "uxnasm: line "; *p; p++) buf[n++] = *p;
  for (int t = line; t >= 10; t /= 10) digits++;
  for (int i = digits - 1; i >= 0; i--, line /= 10) buf[n + i] = (char)('0' + line % 10);
  n += digits;
  buf[n++] = ':'; buf[n++] = ' ';
  for (const char *p = uxnasm_error; *p && n < (int)sizeof buf - 1; p++) buf[n++] = *p;
  buf[n++] = '\n';
  write(1, buf, (unsigned long)n);
}

int tick(void) {
  static Uint8 buttons;
  if (asm_failed) {
    report_asm_error();
    exit(1);
  }
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
  for (;;) {
    long e = __vm_host_call(mouse, 0, 0, 0, 0, 0);
    if (e < 0) break;
    unsigned p = (unsigned)e;
    if ((e >> 32) == 0) varvara_mouse(&u, (p >> 12) & 0xfff, p & 0xfff, (Uint8)(p >> 24));
    else varvara_wheel(&u, (Sint16)(p >> 16), (Sint16)p);
  }
  varvara_screen_vector(&u);
  int w, h;
  const Uint8 *frame = varvara_frame(0, &w, &h);
  if (frame && disp >= 0) __vm_host_call(disp, 0, (long)frame, w, h, 0);
  if (varvara_halted(&u)) exit(0);
  return 0;
}
