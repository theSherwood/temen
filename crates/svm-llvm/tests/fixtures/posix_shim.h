/* A **World-B libc shim** for the LLVM on-ramp: real-C-signature `write`/`read`/`close`/`pipe`/`dup`/
 * `dup2`/`waitpid` + a fork-free `sh_spawn`, each forwarding to the embedder-granted POSIX personality
 * (`svm_run::posix::posix_cap`, resolved by the name `"posix"`) through `__vm_host_call`. This is the
 * on-ramp analogue of the chibicc world's `demos/shell/shim.c` — the layer a real shell links against,
 * adapting C conventions to the personality's op ABI. Op numbers track `crates/svm-posix/src/lib.rs`.
 *
 * A translation-only note (not a runtime one): a program that reaches the host *solely* through
 * `__vm_host_call` never trips svm-llvm's `needs_powerbox_entry`, so it gets no synthesized `_start`.
 * Any real program links libc — the includer calls `printf` (below) — which supplies the `write` import
 * that forces the powerbox entry. */

#ifndef SVM_POSIX_SHIM_H
#define SVM_POSIX_SHIM_H

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
  PX_SPAWN = 27,
  PX_WAITPID = 28,
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
static long __px_slen(const char *s) {
  long n = 0;
  while (s[n])
    n++;
  return n;
}

/* Real libc signatures over the cap — what a shell's own code calls. */
static long write(int fd, const void *buf, long n) { return __px_call(PX_WRITE, fd, (long)buf, n, 0); }
static long read(int fd, void *buf, long n) { return __px_call(PX_READ, fd, (long)buf, n, 0); }
static int close(int fd) { return (int)__px_call(PX_CLOSE, fd, 0, 0, 0); }
static int pipe(int fds[2]) { return (int)__px_call(PX_PIPE, (long)fds, 0, 0, 0); }
static int dup(int fd) { return (int)__px_call(PX_DUP, fd, 0, 0, 0); }
static int dup2(int oldfd, int newfd) { return (int)__px_call(PX_DUP2, oldfd, newfd, 0, 0); }

/* The fork-free spawn primitive: launch a registered command by name (it inherits the caller's fd 0 as
 * stdin and routes its stdout to the caller's fd 1), returning a pid. `waitpid` reaps its status. */
static long sh_spawn(const char *name) { return __px_call(PX_SPAWN, (long)name, __px_slen(name), 0, 0); }
static int waitpid(long pid, int *status, int opts) {
  return (int)__px_call(PX_WAITPID, pid, (long)status, opts, 0);
}

#endif
