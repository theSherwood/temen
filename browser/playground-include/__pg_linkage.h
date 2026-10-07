#ifndef __PG_LINKAGE_H
#define __PG_LINKAGE_H

/* Linkage of the seeded libc's definitions, in the three ways it gets compiled (#1392).
 *
 *   bodies in the program (default) — `static inline` / `static`. This is what makes an unused seeded
 *     function cost nothing: chibicc's `mark_live` pass roots every definition that is not
 *     `static inline`, so external linkage would emit *every* `printf`/`snprintf`/`qsort` body into
 *     every program (measured: +8% IR on a three-call `printf` program). The heap is not among the
 *     bodies (#2172): `malloc` and friends are the heap unit's, so a program that is to run is still
 *     compiled as a unit (`--emit-object`) and linked against it.
 *   the libc unit (`__PG_LIBC_UNIT`, i.e. `__pg_libc.c`) — external linkage, so `--emit-object`
 *     publishes each body in the unit's export table for a program unit to link against.
 *   a program unit (`__PG_LIBC_DECLS_ONLY`) — no bodies at all; the header exposes prototypes and
 *     `extern` data, which the linker resolves to the prebuilt libc unit.
 */
#if defined(__PG_LIBC_DECLS_ONLY) || defined(__PG_LIBC_UNIT)
#define __PG_FN
#define __PG_DATA
#else
#define __PG_FN static inline
#define __PG_DATA static
#endif

#endif
