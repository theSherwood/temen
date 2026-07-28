#ifndef __ERRNO_H
#define __ERRNO_H

// <errno.h> — a single guest-global `errno`. The playground libc is pure computation (no failing
// syscalls), so nothing sets it; it exists so programs that read/clear `errno` compile and run.
static int __pg_errno;
#define errno __pg_errno
#define EDOM 33
#define ERANGE 34
#define EINVAL 22
#define ENOMEM 12

#endif
