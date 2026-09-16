#ifndef __UNISTD_H
#define __UNISTD_H
// <unistd.h> — the playground's **powerbox syscall layer**: the raw capability edge the frontend
// recognizes, plus the POSIX file-descriptor surface built on it. <stdio.h> includes this header for
// the edge rather than declaring it a second time, so there is one definition site per syscall and
// the two co-include cleanly. Also carries inert stubs for the driver-only calls (fork/exec) so
// chibicc.h parses; a cc1 --emit-ir compile uses none of the process calls.
#include <stddef.h>
typedef long ssize_t;
typedef int pid_t;

// ---- the raw powerbox edge (recognized builtins, never real calls) ------------------------------
// `__vm_stream_write(buf, len)` / `__vm_stream_read(buf, len)` lower to `call.sym
// "stream_write"/"stream_read"` on the ambient Stream cap (stdout / stdin). They exist as *distinct
// names* from the fd-less `write`/`read` builtins precisely because this header **defines**
// `write`/`read` to fd-dispatch: chibicc substitutes a builtin only for a name that is declared and
// not defined in the translation unit (`codegen_ir.c`), so the definitions below shadow them — and a
// dispatcher that reached stdout by calling `write` would recurse into itself.
extern long __vm_stream_write(const void *buf, unsigned long n);
extern long __vm_stream_read(void *buf, unsigned long n);

// #1323 (c_interpret #16, file I/O): file access over the powerbox `fs` capability. `__vm_fs(op,…)`
// is the on-ramp fs seam the frontend recognizes (it lowers to `call.sym "vm_fs"`, with the op
// selected by arg0), so one call carries the whole open/read/write/seek/close protocol — the same
// `temen-fs` backend Postgres/chibicc use. The on-ramp mounts a private, in-memory read-write memfs
// (`grant_onramp_caps`). fd 0/1/2 are the ambient Stream; *file* fds (>= 3, minted by FS_OPEN) reach
// the memfs. A program that only writes stdout never touches these.
extern long __vm_fs(long op, long a, long b, long c, long d);
enum { __FS_OPEN = 0, __FS_READ = 1, __FS_WRITE = 2, __FS_SEEK = 3, __FS_CLOSE = 4 };
enum { __FS_O_READ = 1, __FS_O_WRITE = 2, __FS_O_APPEND = 4, __FS_O_TRUNC = 8, __FS_O_CREATE = 16 };
#ifndef SEEK_SET
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#endif

// ---- the POSIX fd surface ----------------------------------------------------------------------
// These are `static inline` in *every* compile mode, unlike the rest of the seeded libc, whose bodies
// move into the prebuilt libc unit (#1392, `__pg_linkage.h`). That is deliberate and `write`/`read`
// force it: they must be **definitions** in whatever unit calls them, or the frontend substitutes
// its fd-less builtins and the fd is silently dropped — a `write(fd, …)` to a file would land on
// stdout. A decls-only program unit is exactly the case where a linked body leaves nothing behind
// but a prototype, so deferring these would break the shape c_interpret and the play card use.
// `open`/`close`/`lseek` follow them for symmetry; all five together are ~20 lines of tokenize,
// noise next to the hundreds this header set already moved out.
static inline int write(int fd, char *buf, long n) {
  if (fd > 2) return (int)__vm_fs(__FS_WRITE, fd, (long)buf, n, 0); // file fd → the memfs
  return (int)__vm_stream_write(buf, (unsigned long)n);             // 0/1/2 → the ambient Stream
}
static inline int read(int fd, char *buf, long n) {
  if (fd > 2) return (int)__vm_fs(__FS_READ, fd, (long)buf, n, 0);
  return (int)__vm_stream_read(buf, (unsigned long)n);
}
// `open`'s flags reach the cap **untranslated**: <fcntl.h> defines the POSIX `O_*` names as the fs
// cap's own bit values, so there is no mapping here to drift out of step (see that header for why
// the values are the cap's and not Linux's). Bits the cap does not define it ignores, and the
// trailing `mode` argument (`0644` &c.) has no meaning in the memfs. Returns a fd >= 3, or -errno.
static inline int open(const char *path, int flags, ...) {
  long n = 0;
  while (path[n]) n++;
  return (int)__vm_fs(__FS_OPEN, (long)path, n, flags, 0);
}
static inline int close(int fd) {
  if (fd <= 2) return 0; // leave the ambient stream fds open, as fclose does
  return (int)__vm_fs(__FS_CLOSE, fd, 0, 0, 0);
}
// SEEK_SET/CUR/END == 0/1/2 == the cap's `whence`.
static inline long lseek(int fd, long off, int whence) {
  return __vm_fs(__FS_SEEK, fd, whence, off, 0);
}

static inline int unlink(const char *p) { (void)p; return 0; }
static inline int access(const char *p, int m) { (void)p; (void)m; return -1; }
static inline pid_t fork(void) { return -1; }
static inline pid_t getpid(void) { return 1; }
static inline int isatty(int fd) { (void)fd; return 0; }
#endif
