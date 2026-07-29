/* Emit-object os bottom edge (SELFHOST_C.md §7, task #20 increment 2b) — the file-descriptor syscalls
 * chibicc's stdio (`chibicc_extra.c`) and `cc1_main.c` (`file_exists` → `stat`) call, for the path where
 * *chibicc itself* compiles the libc. The twin of the on-ramp's `os_shim.c`, but without its svm-llvm
 * intrinsics: emit-object reaches the powerbox by plain `call.sym`, so the two stream syscalls are just
 * **`extern` `write`/`read`** — the linker leaves them as manifest imports the powerbox binds to its
 * out/in `Stream` cap (`default_cap_resolver`: `write`→Stream op1, `read`→Stream op0). `chibicc_extra.c`
 * already declares and drives those for fd 0/1/2, so they are not redefined here.
 *
 * The **fd≥3 file path is the 2c seam.** The reference `default_cap_resolver` binds `write`/`read`/
 * `exit`/`vm_*` but NOT `open`/`close`/`stat`/`lseek`, and a manifest module refuses to start with an
 * unbindable import (`validate_powerbox_manifest`) — so referencing a real fs cap here would make the
 * whole linked cc1 un-runnable under the default powerbox. Until the fs-powerbox harness (memfs + a
 * resolver that binds the fs ops) lands, these are pure-C stubs returning "no such file": the libc then
 * imports only default-bindable caps, so the linked cc1 instantiates and the stdout/`open_memstream`
 * surface runs. `fopen` returns NULL (no file opens yet); reading real source is the 2c increment. */

#include <sys/stat.h>

/* Stream syscalls: emit-object powerbox caps (bound by name), declared — not defined — here.
 * `chibicc_extra.c` carries the identical `extern` and does the fd 0/1/2 dispatch through them. */
extern long write(int fd, const void *buf, unsigned long n);
extern long read(int fd, void *buf, unsigned long n);

/* fd≥3 filesystem — stubbed until the 2c fs cap. Signatures match `chibicc_extra.c`'s externs and the
 * system `<fcntl.h>`/`<sys/stat.h>` prototypes so the guest source type-checks unchanged. */
int open(const char *path, int flags, ...) {
  (void)path;
  (void)flags;
  return -1; /* ENOSYS-equivalent: no filesystem cap bound yet (2c) */
}

int close(int fd) {
  (void)fd;
  return 0;
}

long lseek(int fd, long off, int whence) {
  (void)fd;
  (void)off;
  (void)whence;
  return -1;
}

int stat(const char *path, struct stat *st) {
  (void)path;
  (void)st;
  return -1; /* `cc1_main.c`'s `file_exists` reads this as "absent" until the fs cap (2c) */
}
