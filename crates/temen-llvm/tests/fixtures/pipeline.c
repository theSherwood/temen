/* A **real shell pipeline** — `gen | up` — driven through the World-B libc shim (`posix_shim.h`) over
 * the POSIX personality, on the LLVM on-ramp. This is the plumbing a shell performs for `a | b`, with
 * the personality's fork-free `sh_spawn`: wire stage 1's stdout to a pipe, run it, then wire the pipe to
 * stage 2's stdin and run that — with the classic save/restore of the real stdout across the redirect.
 *
 * The embedder wires the spawn delegate so `gen` emits "hello" and `up` uppercases its stdin. The
 * pipeline therefore lands "HELLO" on the personality's captured stdout; the `printf` marker lands on
 * the powerbox stdout. Regenerate `pipeline.ll` with `clang -O2 -emit-llvm -c -I <fixtures>` + `llvm-dis`. */

#include "posix_shim.h"

int main(void) {
  int p[2];
  if (pipe(p) != 0)
    return 1;
  int r = p[0], w = p[1];

  /* stage 1: `gen` — its stdout goes to the pipe's write end. Save the real stdout first so we can
   * restore it for stage 2's output. */
  int saved_out = dup(1);
  if (saved_out < 0)
    return 2;
  dup2(w, 1);
  close(w);
  int st = 0;
  long pid1 = sh_spawn("gen");
  if (pid1 < 0)
    return 3;
  if (waitpid(pid1, &st, 0) != (int)pid1)
    return 4;

  /* restore the real stdout, then wire the pipe's read end to stage 2's stdin. */
  dup2(saved_out, 1);
  close(saved_out);
  dup2(r, 0);
  close(r);

  /* stage 2: `up` — reads the piped "hello", writes "HELLO" to the (restored) real stdout. */
  long pid2 = sh_spawn("up");
  if (pid2 < 0)
    return 5;
  if (waitpid(pid2, &st, 0) != (int)pid2)
    return 6;

  printf("pipeline ok\n");
  return 0;
}
