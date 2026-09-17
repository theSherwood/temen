/* A minimal **durable** C guest (#1534): a loop that writes one line per iteration through the
 * powerbox `Stream` capability, so every iteration is a `call.cap` the durable transform treats as a
 * may-suspend point and every back-edge carries a freeze poll. Armed with
 * `arm_freeze_after_backedges`, a run of this guest unwinds mid-loop into the shadow arena its
 * module declares (`--shadow-arena`), freezes, and thaws to finish the remaining lines — so the two
 * halves' stdout concatenated equals the uninterrupted run.
 *
 * The committed `durable_probe.ll` is this file compiled with
 * `clang -O1 -S -emit-llvm --target=x86_64-unknown-linux-gnu` (regenerate the same way after
 * editing). `-O1` keeps the loop rolled; the counter lives in a local, which is exactly the state
 * the freeze has to carry across in the shadow frame. */

extern long write(long fd, const char *buf, long n);

int main(void) {
  for (int i = 0; i < 5; i++) {
    write(1, "tick\n", 5);
  }
  return 0;
}
