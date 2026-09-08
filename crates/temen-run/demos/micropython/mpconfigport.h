/* Temen embed-port configuration for MicroPython on the LLVM on-ramp.
 *
 * Starts from the stock embed example's minimal config and flips exactly the switches the on-ramp
 * needs — every one a supported MicroPython feature toggle, so **no MicroPython source is patched**
 * (INVARIANTS: fetched-not-vendored). Each `#define` below is a translator gap closed by config; see
 * README.md "gap-walk".
 */
#include <port/mpconfigport_common.h>

// Minimal starting configuration (disables optional features), plus the compiler + GC we need for a
// real REPL.
#define MICROPY_CONFIG_ROM_LEVEL          (MICROPY_CONFIG_ROM_LEVEL_MINIMUM)
#define MICROPY_ENABLE_COMPILER           (1)
#define MICROPY_ENABLE_GC                 (1)
#define MICROPY_PY_GC                     (1)

// GAP 1 — GC root scan without inline asm. The default `gchelper_generic.c` reads callee-saved
// registers with x86-64 inline asm (`register long rbx asm("rbx")`), which the on-ramp does not
// execute. `MICROPY_GCREGS_SETJMP` uses a portable setjmp-based register capture instead — and Temen
// supports setjmp/longjmp on all three engines.
#define MICROPY_GCREGS_SETJMP             (1)

// GAP 2 — non-local returns without inline asm. The default `nlr*.c` for x86-64 is hand-written asm
// (`nlr_push` does a `movq …; jmp nlr_push_tail`). `MICROPY_NLR_SETJMP` routes MicroPython's exception
// unwinding through setjmp/longjmp — the path the Python plan predicted (MicroPython's `nlr` *is*
// setjmp-shaped) and the one Temen lowers to its core `SetJmp`/`LongJmp` ops.
#define MICROPY_NLR_SETJMP                (1)

// Real Python floats (double), computed with the reused guest libm (openlibm).
#define MICROPY_PY_BUILTINS_FLOAT         (1)
#define MICROPY_FLOAT_IMPL                (MICROPY_FLOAT_IMPL_DOUBLE)

// GAP 3 — no `half` (f16) IR type. With native f16 enabled, `py/binary.c` emits LLVM `half` for the
// array/struct 'e' typecode, which is outside the on-ramp's f64/f32 scope ("Milestone 1+"). Forcing
// the software half-float codec (pure integer ops) keeps floats on while removing the `half` type.
#define MICROPY_FLOAT_USE_NATIVE_FLT16    (0)
