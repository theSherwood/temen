# Uxn on Temen — the Varvara machine as a playground reactor

[Uxn](https://wiki.xxiivv.com/site/uxn.html) is a tiny 8-bit stack machine (32 opcodes × three mode
bits, 64 KiB of memory, two 256-byte stacks); [Varvara](https://wiki.xxiivv.com/site/varvara.html) is its
device layer. This directory runs it in the sandbox as a **reactor guest** — the bounce.c / Doom shape:
`_start` reads the ROM through the `fs` capability, and the page calls `tick()` once per animation
frame, which drains the `keyboard` and `mouse` capabilities into the Controller and Mouse devices,
fires the Screen vector, and presents the composed frame through `display`.

Everything here is **clean-room, written from the public spec** (the reference sources were not
vendored); the CPU was cross-checked against uxn5's spec-compliant core, the assembler byte-for-byte
against uxn5's assembler on this dialect.

## Files

- **`uxn.h` / `uxn.c`** — the CPU. One `switch` over the 5 opcode bits; the mode bits are handled
  uniformly (short widens operands, return swaps the stacks, keep pops from a scratch pointer).
  Circular stacks and memory, so no instruction can fault. Freestanding.
- **`varvara.c`** — the devices a framebuffer-and-keyboard host can serve: **System** (palette,
  expansion fill/copy, debug, halt), **Console** (output), **Screen** (two 2-bit layers, pixel/fill/
  sprite ops with the standard blending table, composed to RGBA), **Controller** (buttons + key),
  **Datetime** (a deterministic virtual clock — the on-ramp grants no wall clock, and determinism is
  what the differential wants), **Mouse** (pointer, buttons, wheel). Audio and File are absent: the
  playground has no such capabilities yet; their ports are inert bytes.
- **`main.c`** — the reactor entry (`#include`s the two above as one translation unit). Maps JS
  keyCodes to the Controller (arrows; Ctrl = A, Alt = B, Shift = Select, Home = Start; letters/digits/
  punctuation/Space/Enter/Backspace/Tab/Esc as the key byte, Shift-aware on the US layout) and the
  packed `mouse` events to the Mouse device. A halted ROM (System/state) exits the guest, which ends
  the reactor loop.
- **`uxn_diff.c`** — the headless **frame-hash differential** (the doom_diff.c shape): reads a ROM
  from stdin, runs N frames under a fixed key script, prints an FNV-1a hash per composed frame. Built
  both as a Temen guest and as a native `cc` binary from this one file; the streams must match
  byte-for-byte. Driven by `crates/temen-llvm/tests/uxn_diff.rs` (skips without clang/cc).
- **`uxnasm.c`** — a small Uxntal assembler (a build-time host tool, not shipped in the guest):
  opcodes with modes, literals, raw hex, padding, labels/sublabels, every reference sigil, lambdas,
  macros, strings, comments. No `~include`.
- **`demo.tal`** — the demo ROM: a 256×192 striped screen, a title set in a 1bpp font, a swarm of
  2bpp sprites bouncing on the foreground layer, a player steered by the arrow keys or placed with a
  click; any letter or a wheel notch cycles the palette, Space resets the swarm.
- **`uxn_corpus.c` + `corpus/`** — the **golden opcode corpus**: 303 programs (random straight-line
  programs over every non-control-flow opcode in every mode, plus hand-written control-flow programs
  and primes) with end states recorded from uxn5's spec-compliant core. `uxn_corpus.c` replays them on
  `uxn.c`; `corpus/gencorpus.mjs` documents how the corpus was produced. Run by
  `crates/temen-llvm/tests/uxn_diff.rs` (`cpu_matches_golden_corpus`).
- **`build.sh`** — assembles the ROM and builds the guest (see the header). The committed assets
  `browser/web/assets/uxn.temen` + `uxn_demo.rom` come from `ONLY=uxn bash scripts/rebuild-assets.sh`.

## Running

```sh
sh crates/temen-run/demos/uxn/build.sh            # → /tmp/temen_uxn_cache/{uxn.temen,uxn_demo.rom}
cargo test -p temen-browser --test uxn_reactor    # the reactor wiring over the committed assets
cargo test -p temen-llvm --test uxn_diff          # native vs guest frame hashes (needs clang + cc)
```

To run your own ROM in the playground, pick it in the card's ROM picker or drop it on the canvas (the
guest opens whatever the page serves as `boot.rom`); to write one, assemble it with `uxnasm` from this
directory. ROMs that need the Audio or File devices run with those devices inert.
