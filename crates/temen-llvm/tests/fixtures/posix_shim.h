/* A **World-B libc shim** for the LLVM on-ramp: real-C-signature `write`/`read`/`close`/`pipe`/`dup`/
 * `dup2`/`waitpid` + `sh_spawn` (a `posix_spawn`), each forwarding to the embedder-granted POSIX personality
 * (`temen_run::posix::posix_cap`, resolved by the name `"posix"`) through `__vm_host_call`. This is the
 * on-ramp analogue of the chibicc world's `demos/shell/shim.c` — the layer a real shell links against,
 * adapting C conventions to the personality's op ABI. Op numbers track `crates/temen-posix/src/lib.rs`.
 *
 * A translation-only note (not a runtime one): a program that reaches the host *solely* through
 * `__vm_host_call` never trips temen-llvm's `needs_powerbox_entry`, so it gets no synthesized `_start`.
 * Any real program links libc — the includer calls `printf` (below) — which supplies the `write` import
 * that forces the powerbox entry. */

#ifndef TEMEN_POSIX_SHIM_H
#define TEMEN_POSIX_SHIM_H

extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);
int printf(const char *, ...);

/* Personality op numbers (POSIX.md ABI table). */
enum {
  PX_WRITE = 0,
  PX_READ = 1,
  PX_CLOSE = 6,
  PX_PIPE = 23,
  PX_DUP2 = 24,
  PX_DUP = 25,
  PX_WAITPID = 28,
  PX_PSPAWN = 62,
};

static int __px_handle = -1;
static int __px(void) {
  if (__px_handle < 0)
    __px_handle = __vm_cap_resolve("posix", 5);
  return __px_handle;
}
static long __px_call(int op, long a, long b, long c, long d) {
  return __vm_host_call(__px(), op, a, b, c, d);
}

/* Real libc signatures over the cap — what a shell's own code calls. */
static long write(int fd, const void *buf, long n) { return __px_call(PX_WRITE, fd, (long)buf, n, 0); }
static long read(int fd, void *buf, long n) { return __px_call(PX_READ, fd, (long)buf, n, 0); }
static int close(int fd) { return (int)__px_call(PX_CLOSE, fd, 0, 0, 0); }
static int pipe(int fds[2]) { return (int)__px_call(PX_PIPE, (long)fds, 0, 0, 0); }
static int dup(int fd) { return (int)__px_call(PX_DUP, fd, 0, 0, 0); }
static int dup2(int oldfd, int newfd) { return (int)__px_call(PX_DUP2, oldfd, newfd, 0, 0); }

/* `posix_spawn` with no file actions: start the program at `path` as a new process, which inherits the
 * caller's descriptors (fd 0 its stdin, fd 1 its stdout), returning its pid. `waitpid` reaps its
 * status. The request is `{path, argv, envp, actions, nactions}`. */
static long sh_spawn(const char *path) {
  long req[5] = {(long)path, 0, 0, 0, 0};
  return __px_call(PX_PSPAWN, (long)req, 0, 0, 0);
}
static int waitpid(long pid, int *status, int opts) {
  return (int)__px_call(PX_WAITPID, pid, (long)status, opts, 0);
}

#endif
