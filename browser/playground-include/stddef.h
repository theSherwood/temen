#ifndef __STDDEF_H
#define __STDDEF_H

// <stddef.h> — the core typedefs/macros. `size_t` is also defined (guarded the same way) by the
// other playground headers, so including several together is fine.
typedef unsigned long size_t;
typedef long ptrdiff_t;
#ifndef NULL
#define NULL ((void *)0)
#endif
#define offsetof(type, member) ((size_t)&(((type *)0)->member))

#endif
