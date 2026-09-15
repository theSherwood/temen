/* **Differential self-test for `nifler_shim.c`'s `strtod`, against the host libc.**
 *
 * nifler bakes every parsed float straight into the `.p.nif` it emits, so a last-place error here is
 * a compiler that silently disagrees with native nimony about what a constant means — and the asset
 * lane's oracle diff (step 5) would only catch it for literals that happen to appear in `inputs/`.
 * This checks the routine directly over the shapes that actually break naive implementations:
 * the constants `std/fenv` and `std/math` contain, every power of ten across the whole exponent
 * range, exact halfway cases at the 53-bit boundary, subnormals (where rounding must happen at a
 * *shorter* significand, or it double-rounds), and 500-digit strings.
 *
 * Built and run by `build_nifler_temen.sh` step 0 with the host's `strtod` as the oracle. The shim is
 * `#include`d rather than linked so the test sees the same code the guest gets; the guard below keeps
 * the rest of the shim (which needs the guest's on-ramp builtins) out. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NIFLER_SHIM_STRTOD_ONLY 1
#include "shim_strtod_extract.h" /* the strtod block, carved out by the build script */

static long bad = 0, n = 0;

static void check(const char *s) {
  double a = nim_strtod(s, 0), b = strtod(s, 0);
  uint64_t ua, ub;
  memcpy(&ua, &a, 8);
  memcpy(&ub, &b, 8);
  n++;
  if (ua != ub) {
    if (bad < 20)
      printf("  MISMATCH %-34s ours=%.17g libc=%.17g\n", s, a, b);
    bad++;
  }
}

int main(void) {
  /* Real constants from std/fenv and std/math, plus the boundaries. */
  static const char *fixed[] = {
      "1.17549435e-38", "1.19209290e-07", "2.2250738585072014E-308", "2.2204460492503131E-16",
      "1.7976931348623157E+308", "1.5e100", "1.5e-100", "2.2204460492503131", "1.5", "0", "0.0",
      "1e22", "1e23", "1e-22", "1e-23", "9007199254740993", "4.9406564584124654e-324",
      "1e308", "1e309", "1e-308", "1e-320", "1e-324", "1e-325", "0.1", "0.2", "0.3",
      "3.141592653589793", "2.718281828459045", "5e-324", "2.5e-324", "7.4e-324",
      "0.000000000000000000000000000001", "1234.5678e-300", "123456789012345678901234567890",
  };
  for (unsigned i = 0; i < sizeof fixed / sizeof *fixed; i++) check(fixed[i]);

  /* Every power of ten, and a near-max / near-min significand at each. */
  char b[600];
  for (int e = -330; e <= 309; e++) {
    snprintf(b, sizeof b, "1e%d", e); check(b);
    snprintf(b, sizeof b, "9.999999999999999e%d", e); check(b);
    snprintf(b, sizeof b, "4.9406564584124654e%d", e); check(b);
  }

  /* Round-trip every bit pattern we can reach cheaply: %.17g is exact, %.30g pads past the tie. */
  unsigned long long st = 88172645463325252ULL;
  for (int i = 0; i < 60000; i++) {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    unsigned long long bits = (st % 0x7FE0000000000000ULL) + 1;
    double v;
    memcpy(&v, &bits, 8);
    snprintf(b, sizeof b, "%.17g", v); check(b);
    snprintf(b, sizeof b, "%.30g", v); check(b);
  }

  /* Random significand x exponent, including the subnormal range. */
  for (int i = 0; i < 120000; i++) {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    unsigned long long m = st % 1000000000000000000ULL;
    int e = (int)((st >> 32) % 700) - 350;
    int digits = (int)((st >> 20) % 18) + 1;
    char mb[32];
    snprintf(mb, sizeof mb, "%0*llu", digits, m);
    snprintf(b, sizeof b, "%c%s.%se%d", (st & 1) ? '-' : '+', mb, mb, e);
    check(b);
  }

  /* A 500-digit string — the buffer `parseBiggestFloat` hands us on its slow path. */
  memset(b, '0', sizeof b);
  b[0] = '1'; b[1] = '.';
  for (int i = 2; i < 520; i++) b[i] = (char)('0' + (i % 10));
  b[520] = 0;
  check(b);
  strcat(b, "e-300");
  check(b);

  printf("  strtod self-test: %ld cases, %ld mismatches vs host libc\n", n, bad);
  return bad != 0;
}
