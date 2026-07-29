/* Probe for the **POSIX personality reached from the LLVM on-ramp** (svm-run's `posix::posix_cap`):
 * resolve the embedder-granted personality by name (`__vm_cap_resolve("posix")`, §7 cap.self.resolve)
 * and drive the process/fd ABI through `__vm_host_call` (§7 host-defined capability) — the same idiom
 * `fs_probe.c` uses for the `fs` cap. Exercises the ops added for a real shell: `pipe`/`dup2` (the fd
 * surface), `posix_spawn`/`waitpid` (the fork-free process model, with fd inheritance), and `write`.
 * Exit 0 iff every step behaved; any failure returns its step number.
 *
 * The committed `posix_probe.ll` is this file compiled with `clang -O2 -emit-llvm -S` (regenerate the
 * same way after editing).
 *
 * The embedder wires `Posix::set_spawn` to an uppercasing delegate that exits 42; `spawn("up")` inherits
 * this guest's fd 0 (preloaded stdin) as the child's input and routes the child's stdout to fd 1 (the
 * personality's captured stdout), so `posix.stdout()` observes the child's `"HELLO"` followed by this
 * guest's own `"ok"` marker — the Rust test asserts both that and `WEXITSTATUS == 42`. */

extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);
/* A standard libc call, so the translator synthesizes the powerbox `_start` entry (svm-llvm's
 * `needs_powerbox_entry`): a program that reaches the host *only* through `__vm_host_call` has no
 * import to trip that, and its `main` would be func 0 with an unfilled data-SP param. `printf` of a
 * constant lowers to `puts` → the `write` powerbox import, exactly as `fs_probe.c` does; this marker
 * therefore lands on the *powerbox* stdout (the run's `stdout`), distinct from the personality's. */
extern int printf(const char *, ...);

enum { WRITE = 0, READ = 1, PIPE = 23, DUP2 = 24, SPAWN = 27, WAITPID = 28 };

static int px;
static long hc(int op, long a, long b, long c, long d) { return __vm_host_call(px, op, a, b, c, d); }

int main(void) {
  px = __vm_cap_resolve("posix", 5);
  if (px < 0) return 1;

  /* 1. pipe(): the read/write fds land in the guest buffer. */
  int fds[2];
  if (hc(PIPE, (long)fds, 0, 0, 0) != 0) return 2;
  int r = fds[0], w = fds[1];

  /* 2. dup2 the write end onto a high fd; a write through the dup reaches the same buffer. */
  if (hc(DUP2, w, 9, 0, 0) != 9) return 3;
  if (hc(WRITE, 9, (long)"ping", 4, 0) != 4) return 4;
  char buf[8];
  if (hc(READ, r, (long)buf, 8, 0) != 4) return 5;
  if (buf[0] != 'p' || buf[1] != 'i' || buf[2] != 'n' || buf[3] != 'g') return 6;

  /* 3. posix_spawn a child (the delegate uppercases the inherited stdin, exits 42); waitpid its status.
   *    fd 0 (preloaded "hello") is the child's stdin; the child's stdout follows fd 1 to the captured
   *    stdout the test reads back. */
  long pid = hc(SPAWN, (long)"up", 2, 0, 0);
  if (pid < 0) return 7;
  int st = 0;
  if (hc(WAITPID, pid, (long)&st, 0, 0) != pid) return 8;
  if (((st >> 8) & 255) != 42) return 9;

  /* 4. this guest's own marker — on the *powerbox* stdout (the run's stdout), distinct from the
   *    personality's captured stdout that holds the child's "HELLO". */
  printf("posix probe ok\n");
  return 0;
}
