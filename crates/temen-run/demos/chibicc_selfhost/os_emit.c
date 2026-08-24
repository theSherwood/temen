/* Emit-object os bottom edge (SELFHOST_C.md §7, task #21 increment 2c) — the file-descriptor syscalls
 * chibicc's stdio (`chibicc_extra.c`) and `cc1_main.c` (`file_exists` → `stat`) call, for the path where
 * *chibicc itself* compiles the libc. The twin of the on-ramp's `os_shim.c`, but without its temen-llvm
 * intrinsics: emit-object reaches the powerbox through the recognized `__vm_*` builtins the frontend
 * lowers to `call.sym` (`codegen_ir.c`), which the host binds to caps at load. Two edges:
 *
 *   * **Stream** (fd 0/1/2) — `__vm_stream_write`/`__vm_stream_read` → the powerbox `Stream` cap
 *     (stdout/stderr/stdin). Distinct names from the `write`/`read` *builtins*, because this file
 *     *defines* `write`/`read` (to fd-dispatch), which shadows those builtins — so the shim reaches the
 *     raw stream through `__vm_stream_*` without recursing into its own definition.
 *   * **Filesystem** (fd≥3) — `__vm_fs(op, …)` → one manifest slot carrying the whole fs op protocol
 *     (`crates/temen-run/src/fs.rs`), op selected by arg0, the host binds it to a seeded memfs. This is
 *     what lets the linked compiler `fopen` and read a real source / `#include` header in the sandbox.
 *
 * `write`/`read` here fd-dispatch (0/1/2 → Stream, ≥3 → fs), exactly like `os_shim.c` does for the
 * on-ramp; `open`/`close`/`lseek`/`stat` are the fs cap. */

#include <sys/stat.h>

/* Powerbox edges — recognized `__vm_*` builtins (declared, intercepted by the frontend; never real
 * calls). The op protocol constants mirror `crates/temen-run/src/fs.rs`. */
extern long __vm_stream_write(const void *buf, unsigned long n); /* → Stream op1 (stdout/stderr) */
extern long __vm_stream_read(void *buf, unsigned long n);        /* → Stream op0 (stdin) */
extern long __vm_fs(long op, long a, long b, long c, long d);    /* fs cap; op in arg0 */

enum { FS_OPEN = 0, FS_READ, FS_WRITE, FS_SEEK, FS_CLOSE, FS_STAT_OP = 14 };
enum { FS_O_READ = 1, FS_O_WRITE = 2, FS_O_APPEND = 4, FS_O_TRUNC = 8, FS_O_CREATE = 16 };

extern unsigned long strlen(const char *); /* mem_shim.c (same unity TU) */

/* The fs cap is rooted (it rejects absolute paths and `..` — the confinement seam in fs.rs), but chibicc
 * opens guest-absolute paths (`/in.c`, `/include/foo.h`). Strip leading slashes so "/" *is* the cap root;
 * confinement is unchanged (the cap still forbids `..`). A path that collapses to "" becomes ".". */
static const char *fs_norm(const char *path) {
  if (!path)
    return path;
  const char *p = path;
  while (*p == '/')
    p++;
  return *p ? p : ".";
}

/* ---- stream + file read/write: fd-dispatch (0/1/2 → Stream, ≥3 → fs cap) --------------------- */
long write(int fd, const void *buf, unsigned long n) {
  if (fd == 1 || fd == 2)
    return __vm_stream_write(buf, n);
  return __vm_fs(FS_WRITE, fd, (long)buf, (long)n, 0);
}
long read(int fd, void *buf, unsigned long n) {
  if (fd == 0)
    return __vm_stream_read(buf, n);
  return __vm_fs(FS_READ, fd, (long)buf, (long)n, 0);
}

/* ---- files ---------------------------------------------------------------------------------- */
/* `open` receives system `O_*` octal flags (chibicc_extra.c's `mode_flags` builds them: 0=RDONLY,
 * 01=WRONLY, 0100=CREAT, 01000=TRUNC, 02000=APPEND). Map to the fs cap's `FS_O_*` bitset. */
int open(const char *path, int flags, ...) {
  const char *p = fs_norm(path);
  long f = 0;
  int acc = flags & 03; /* O_ACCMODE */
  if (acc == 0 || acc == 02)
    f |= FS_O_READ; /* RDONLY or RDWR */
  if (acc == 01 || acc == 02)
    f |= FS_O_WRITE; /* WRONLY or RDWR */
  if (flags & 02000)
    f |= FS_O_APPEND;
  if (flags & 01000)
    f |= FS_O_TRUNC;
  if (flags & 0100)
    f |= FS_O_CREATE;
  long rc = __vm_fs(FS_OPEN, (long)p, (long)strlen(p), f, 0);
  return (int)rc; /* fd ≥ 0, or negative errno the caller treats as failure (< 0) */
}

int close(int fd) { return (int)__vm_fs(FS_CLOSE, fd, 0, 0, 0); }

long lseek(int fd, long off, int whence) {
  /* SEEK_SET/CUR/END == 0/1/2 == the cap's whence. */
  return __vm_fs(FS_SEEK, fd, whence, off, 0);
}

/* `stat` fills the fs cap's fixed 72-byte little-endian StatBuf (fs.rs::stat_bytes: 0:mode 8:size
 * 16:mtime_s …). cc1's `file_exists` only checks the return code; `__TIMESTAMP__` reads `st_mtime`
 * (stubbed to the 1970 epoch by the cap). Copy the fields the frontend can observe. */
int stat(const char *path, struct stat *st) {
  const char *p = fs_norm(path);
  unsigned char b[72];
  long rc = __vm_fs(FS_STAT_OP, (long)p, (long)strlen(p), (long)b, 72);
  if (rc != 0)
    return -1;
  for (unsigned i = 0; i < sizeof *st; i++)
    ((char *)st)[i] = 0;
  unsigned mode = (unsigned)b[0] | (unsigned)b[1] << 8 | (unsigned)b[2] << 16 | (unsigned)b[3] << 24;
  long size = 0, mtime = 0;
  for (int i = 0; i < 8; i++) {
    size |= (long)b[8 + i] << (8 * i);
    mtime |= (long)b[16 + i] << (8 * i);
  }
  st->st_mode = mode;
  st->st_size = size;
  st->st_mtime = mtime;
  return 0;
}
