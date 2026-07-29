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
 *      `__SVM_STDLIB_H`, which gates out `chibicc_extra.c`'s own `free`/`calloc` (a redefinition
 *      otherwise). So on emit-object the heap is the header's; on the on-ramp it is svm-llvm's synth
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
#define SVM_GUEST 1
#include <stdlib.h>                  /* bundled: static malloc/free/calloc/realloc; defines __SVM_STDLIB_H */
#include "../postgres/shim_errno.h"  /* errno cell + __errno_location */
#include "../postgres/mem_shim.c"    /* memcpy/memset/memmove/memcmp/strcmp/strncmp/strlen */
#include "../postgres/libc_shim.c"   /* __ctype_b_loc/strncpy/strstr/strdup/strtol/strtoul/atoi */
#include "../strtod/strtod.c"        /* correctly-rounded strtod (float literal parse) */
#include "chibicc_extra.c"           /* stdio+memstream, strchr/memchr/strndup/…, time (no allocator) */

/* NOT YET INCLUDED — the two bottom-edge shims are on-ramp-only: `os_shim.c` and `printf_shim.c` call
 * svm-llvm intrinsics (`__vm_stream_write`/`__vm_host_call`/`__vm_cap_resolve`, `__vm_fmt_gen`/…) that
 * chibicc's own `--emit-object` codegen does *not* recognize as builtins (it knows `__vm_map`/`__vm_jit_`
 * /… — codegen_ir.c `scan_caps` — but not those). The emit-object regime instead invokes the host by
 * plain `call.sym "write"`/`"read"`/… that the powerbox binds. So this aggregator needs **emit-object**
 * variants of both — an os bottom edge (fd-dispatch over `write`/`read` [Stream cap] and an fs cap for
 * fd≥3) and a `__vm_fmt_*`-free `vfprintf` (the `%d/%s/%x/%ld/%02d/%.*s/%+ld` surface chibicc uses;
 * `%e`/`%.17g` float emission needs a guest dtoa, but is unexercised on integer inputs). Those land in
 * the next slice; here the reusable, intrinsic-free core (allocator from the bundled header, mem/str,
 * ctype, errno, strtod, and the fd/memory-stream stdio) compiles under emit-object on its own. */
