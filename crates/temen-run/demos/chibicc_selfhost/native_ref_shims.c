/* Native-reference-only shims for the self-host differential (run_selfhost_diff.sh).
 *
 * The reference `chibicc_ref` is built with -mlong-double-64 to match the guest (Appendix A F3:
 * fp80 doesn't lower through the on-ramp, so the guest bounds `long double` to `double`). But the
 * SYSTEM libc's `strtold` still uses the 80-bit x87 return ABI — calling it while `long double` is
 * 64-bit reads back garbage (every float literal came out as the same denormal, 4.68e-310). The
 * guest never hits this: its `strtold` forwards to `strtod` (chibicc_extra.c). Give the native
 * reference the identical forwarding so BOTH sides parse float literals through the same
 * correctly-rounded `strtod` — making the differential a true apples-to-apples comparison of the
 * emitted IR (including the %.17g path). Linked only into chibicc_ref, never into the guest.
 */
#include <stdlib.h>

long double strtold(const char *nptr, char **endptr) {
  return (long double)strtod(nptr, endptr);
}
