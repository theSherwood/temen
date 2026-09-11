// The **libc unit** for the chibicc card (#1392): the one translation unit that carries the seeded
// playground headers' function bodies, compiled once with `chibicc --emit-object` into a linkable
// unit. A user's program is compiled against the same headers while force-including
// `__pg_decls_only.h`, so it sees prototypes only and its compile drops from seconds to
// milliseconds; the calls resolve cross-unit at link time (`temen_link_run` / `temen_link_text`).
//
// `__PG_LIBC_UNIT` gives the bodies external linkage (see `__pg_linkage.h`) so `--emit-object`
// exports them; without it they are `static inline` and would be dead-stripped as unreferenced.
// Keep the include list in step with the headers that carry a `__PG_LIBC_DECLS_ONLY` body guard.

#define __PG_LIBC_UNIT 1

#include <stdio.h>
#include <stdlib.h>
