/* chibicc self-host guest libc — the aggregating -DSVM_GUEST driver (SELFHOST_C.md Appendix B).
 * Mirrors demos/postgres/pg_shims.c: one translation unit that #includes the proven, byte-exact
 * Postgres shims (reused in-place for the first bring-up — the "second consumer" dedup to a shared
 * dir is a follow-up, §5 B open decision) plus chibicc_extra.c for the chibicc-specific remainder.
 * Include order matches the shims' stated dependencies (stdio/printf after os; strtod before its
 * users). Only the shims chibicc actually needs are pulled — no locale/scanf/ipc/event/proc — to
 * keep the undefined-extern surface minimal.
 */
#define SVM_GUEST 1
#include "../postgres/shim_errno.h"  /* errno cell + __errno_location */
#include "../postgres/mem_shim.c"    /* memcpy/memset/memmove/memcmp/strcmp/strncmp/strlen */
#include "../postgres/os_shim.c"     /* open/read/write/close/lseek/stat + fd-dispatch over fs cap */
#include "../postgres/libc_shim.c"   /* __ctype_b_loc/strncpy/strstr/strdup/strtol/strtoul/atoi */
#include "../strtod/strtod.c"        /* correctly-rounded strtod (on-ramp's is a trap stub) */
#include "chibicc_extra.c"           /* allocator, stdio+memstream, strchr/memchr/strndup/…, time */
#include "../postgres/printf_shim.c" /* fprintf/vfprintf/snprintf/vsnprintf — floats via __vm_fmt_* */
