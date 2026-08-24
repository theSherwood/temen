/* chibicc self-host guest libc — the **emit-object** aggregator (SELFHOST_C.md §7, task #20).
 *
 * The twin of `chibicc_libc.c`, but for the path where *chibicc itself* compiles the libc
 * (`--emit-object`) instead of the LLVM on-ramp (clang). Same reused-in-place Postgres shims +
 * `chibicc_extra.c`; two differences the emit-object regime forces:
 *
 *   1. The **allocator comes from chibicc's bundled `<stdlib.h>`** (`frontend/chibicc/include/`),
 *      which defines `static` `malloc`/`free`/`calloc`/`realloc` over the `__vm_map`-growable window
 *      heap — chibicc searches its bundled include dir first, so a plain `#include <stdlib.h>` here
 *      resolves to it (not the system header the on-ramp's clang sees). Including it *first* defines
 *      `__TEMEN_STDLIB_H`, which gates out `chibicc_extra.c`'s own `free`/`calloc` (a redefinition
 *      otherwise). So on emit-object the heap is the header's; on the on-ramp it is temen-llvm's synth
 *      + `chibicc_extra.c` — one source, both regimes.
 *   2. Nothing else changes: `os_shim.c`'s fd-dispatch (0/1/2 → powerbox Stream, ≥3 → fs cap),
 *      `mem_shim`/`libc_shim` string+ctype, `strtod` for float parse, `printf_shim` for the stdio
 *      formatters, `chibicc_extra` for stdio + `open_memstream` + the string/time remainder.
 *
 * Compiled with the self-host prelude force-included (`selfhost_prelude.h`) for the `strtoul`/`atoi`/
 * `strtold` glibc-header gap. The undefined externs that survive into the linked program are exactly
 * the host capabilities the powerbox binds (`write`/`read`/`open`/`close`/`lseek`/`stat` +
 * `vm_map`/`vm_page_size` from the header's heap) — no other libc.
 */
#define TEMEN_GUEST 1
#define __TEMEN_LIBC_OWNER 1           /* this unit holds the ONE shared allocator state (2c); see stdlib.h */
#include <stdlib.h>                  /* bundled: static malloc/free/calloc/realloc; defines __TEMEN_STDLIB_H */
#include "../postgres/shim_errno.h"  /* errno cell + __errno_location */
#include "../postgres/mem_shim.c"    /* memcpy/memset/memmove/memcmp/strcmp/strncmp/strlen */
#include "../postgres/libc_shim.c"   /* __ctype_b_loc/strncpy/strstr/strdup/strtol/strtoul/atoi */
#include "../strtod/strtod.c"        /* correctly-rounded strtod (float literal parse) */
#include "os_emit.c"                 /* emit-object os bottom edge: extern write/read (caps) + fs stubs (2c) */
#include "chibicc_extra.c"           /* stdio+memstream, strchr/memchr/strndup/…, time (no allocator) */
#include "printf_emit.c"             /* __vm_fmt_*-free vfprintf/fprintf/printf/snprintf/vsnprintf */

/* The two bottom-edge shims above are the **emit-object** replacements for the on-ramp's `os_shim.c`
 * and `printf_shim.c`, which reach the host through temen-llvm intrinsics (`__vm_stream_write`/
 * `__vm_host_call`/`__vm_cap_resolve`; `__vm_fmt_*`) that chibicc's own `--emit-object` codegen does not
 * lower. `os_emit.c` invokes the powerbox by plain `extern` `write`/`read` (bound by name to the Stream
 * cap) and stubs the fd≥3 fs path until the 2c fs cap; `printf_emit.c` supplies the float formatters in
 * guest C. The undefined externs that survive into the linked program are then only default-powerbox-
 * bindable caps (`write`/`read`/`exit`/`vm_map`/`vm_page_size`). */
