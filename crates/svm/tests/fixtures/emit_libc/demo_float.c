// Consumer that exercises the **emit-object libc's correctly-rounded float path** (task #22): the
// `printf_emit.c` Steele & White scaled-bignum dtoa behind `%g`/`%e`/`%f`. chibicc emits float literals
// with `%.17g` (codegen_ir.c), so the headline requirement is that `%.17g` prints the 17-significant-
// digit decimal that re-parses a `double` losslessly — including the values that are *not* their own
// short decimal (0.1, 2/3, 1e-7). Linked against `emit_libc.c` and run through `_start` on interp==JIT
// under the powerbox; the stdout bytes are a fixed oracle matching glibc (see the test).
//
// Declares only what it calls (no system headers) so it compiles as a standalone emit-object unit.
int printf(const char *, ...);

int main(void) {
  // %.17g — the round-trip path chibicc itself emits. Values chosen so the correct answer differs
  // from a naive/lossy formatter: 0.1 and 2/3 are non-terminating in binary; -1e-7 shows its true
  // stored value; 6.022e23 needs the exponent form; -0.0 keeps its sign.
  printf("%.17g\n", 0.1);
  printf("%.17g\n", 1.0);
  printf("%.17g\n", 3.14159265358979);
  printf("%.17g\n", 2.0 / 3.0);
  printf("%.17g\n", 1e20);
  printf("%.17g\n", -1e-7);
  printf("%.17g %.17g\n", 123456789.0, 6.022e23);

  // %g trailing-zero stripping + the e-vs-f switchover at exponent -4.
  printf("[%g][%g][%g][%g]\n", 0.5, 100.0, 0.0001, 1e-5);

  // %e / %E fixed fractional width (no stripping) and the uppercase spelling.
  printf("[%e][%E]\n", 12345.678, 0.00042);

  // %f fixed fractional places, including the sub-precision rounding corner (%.3f of 0.007).
  printf("[%.2f][%.0f][%.3f]\n", 3.14159, 2.5, 0.007);

  // Sign/space flags and -0.0 through the same paths.
  printf("[%+.1f][% .1e][%.17g]\n", 4.0, -50.0, -0.0);

  return 0;
}
