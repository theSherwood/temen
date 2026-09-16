#ifndef __FCNTL_H
#define __FCNTL_H
// <fcntl.h> — the `open` flags for the seeded playground libc, mapped onto the **fs cap's** values.
//
// `unistd.h` already declares `open`/`read`/`write`/`close` over the `vm_fs` cap, so the syscall path
// was wired; this header is what the flag names were missing from. Without it, a program that does the
// ordinary thing —
//
//     #include <fcntl.h>
//     int fd = open("f.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);
//
// — fails at the *preprocessor*, with `fcntl.h: cannot open file`, before any of the interesting parts
// are reached (c_interpret#28 cluster F: that was 10 spec failures across `mmap-file`, `posix-fd` and
// `framebuffer`, all of which looked like capability gaps and were none).
//
// **The values are the fs cap's, not Linux's**, and they must stay that way — `temen-fs` decodes the
// flags it defines (`crates/temen-fs/src/lib.rs`: `O_READ` 1, `O_WRITE` 2, `O_APPEND` 4, `O_TRUNC` 8,
// `O_CREATE` 16). Copying glibc's numbers here would compile and then misbehave silently, and one case
// is worth spelling out: on Linux `O_RDONLY` is **0**, a no-op bit pattern that means "no write bits
// set". The fs cap instead gates `readable: flags & O_READ != 0`, so a zero `O_RDONLY` produces a fd
// that is open but **not readable**, and the guest's first `read` returns nothing with no diagnostic.
// Read access here is something you request, not something you get by omission.
//
// Only the flags the seeded libc can honor are defined. A name that has no meaning to the fs cap is
// deliberately absent rather than defined-as-zero, so a program using it fails to compile — visibly —
// instead of being quietly ignored at runtime.

// Access mode. Unlike POSIX, these are independent bits and `O_RDONLY` is *not* zero.
#define O_RDONLY 1 /* temen-fs O_READ */
#define O_WRONLY 2 /* temen-fs O_WRITE */
#define O_RDWR   3 /* O_READ | O_WRITE */

// Creation / status flags.
#define O_APPEND 4  /* temen-fs O_APPEND */
#define O_TRUNC  8  /* temen-fs O_TRUNC  */
#define O_CREAT  16 /* temen-fs O_CREATE */

// `open` itself is declared in <unistd.h> alongside the other fs-cap syscalls; include it so
// `#include <fcntl.h>` alone is enough to call `open`, as it is on a real system.
#include <unistd.h>

#endif
