/* Uxn — the CPU (https://wiki.xxiivv.com/site/uxn.html). A clean-room, spec-based core: 64 KiB of
 * addressable memory (in `UXN_BANKS` 64 KiB banks; the CPU only ever sees bank 0, the System device's
 * expansion port reaches the rest), two 256-byte circular stacks, and 32 base opcodes × three mode
 * bits (short `2`, return `r`, keep `k`). Devices are the 256-byte `dev` page: `DEI`/`DEO` call the
 * emulator's `uxn_dei` / `uxn_deo` (varvara.c) with the port address. Freestanding: no libc. */
#ifndef UXN_H
#define UXN_H

typedef unsigned char Uint8;
typedef signed char Sint8;
typedef unsigned short Uint16;
typedef signed short Sint16;

#define UXN_BANKS 16 /* 1 MiB total: bank 0 is the CPU's memory, banks 1..15 expansion-only */

typedef struct {
  Uint8 dat[256];
  Uint8 ptr; /* index of the next free cell (so `ptr` is also the depth); wraps like the spec says */
} Stack;

typedef struct {
  Uint8 *ram;     /* UXN_BANKS * 0x10000 bytes */
  Uint8 dev[256]; /* the device page */
  Stack wst, rst;
} Uxn;

/* Run from `pc` until BRK (returns 1) or a jump to address 0 (returns 0). A vector is skipped (returns
 * 0) once System/state is set — the emulator checks halting between vectors, the CPU never mid-run. */
int uxn_eval(Uxn *u, Uint16 pc);

/* Device hooks the emulator provides. `uxn_deo` is called AFTER `dev[port]` holds the new byte. */
Uint8 uxn_dei(Uxn *u, Uint8 port);
void uxn_deo(Uxn *u, Uint8 port);

#define PEEK2(p) ((Uint16)((p)[0] << 8 | (p)[1]))
#define POKE2(p, v) ((p)[0] = (Uint8)((v) >> 8), (p)[1] = (Uint8)(v))

#endif
