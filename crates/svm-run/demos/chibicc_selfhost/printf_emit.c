/* Emit-object printf (SELFHOST_C.md §7, task #20 increment 2b) — the varargs `vfprintf`/`fprintf`/
 * `printf`/`snprintf`/`vsnprintf` family for the path where *chibicc itself* compiles the libc.
 *
 * The on-ramp's `printf_shim.c` engine is already pure C for the whole integer/string/char/pointer +
 * width/precision/flags surface chibicc uses (`%d`/`%s`/`%x`/`%ld`/`%02d`/`%.*s`/`%+ld`/`%*s`); its
 * *only* on-ramp dependency is the float path, which delegates to the svm-llvm intrinsics
 * `__vm_fmt_{fix,sci,gen}` (a correctly-rounded bignum dtoa) that chibicc's `--emit-object` codegen does
 * not lower. So this file **defines those three as guest C** and then includes the shim verbatim — one
 * engine, all formats, no intrinsics. (The shim declares them `extern`; defining them first in the same
 * TU satisfies that.)
 *
 * The guest float path is **best-effort, not correctly-rounded**: it prints plausible decimals via
 * `double` arithmetic, good enough that nothing crashes and typical values read correctly, but it does
 * *not* guarantee the 17-significant-digit round-trip `%.17g` needs to re-parse a float constant
 * losslessly. chibicc emits float literals with `%.17g` (codegen_ir.c), so a program with float
 * literals needs the correctly-rounded dtoa — that is the float-input increment (port the bignum dtoa
 * to guest C). On the integer inputs 2b targets, the float path is never reached at run time; it exists
 * so the module links and instantiates. */

#include <stdarg.h>
#include <stddef.h>

/* Emit the leading sign per the printf flag bits (bit1 '+', bit2 ' '), consuming a negative (or -0.0)
 * sign from x. Returns bytes written. */
static int svm_fmt_sign(char *o, double *x, int flags) {
  int n = 0;
  int neg = (*x < 0.0) || (*x == 0.0 && 1.0 / *x < 0.0); /* -0.0 → 1/x is -inf */
  if (neg) {
    o[n++] = '-';
    *x = -*x;
  } else if (flags & 2) {
    o[n++] = '+';
  } else if (flags & 4) {
    o[n++] = ' ';
  }
  return n;
}

/* inf/nan without <math.h>. `up` selects the upper-case spelling (bit3 of the flags the shim passes). */
static int svm_fmt_special(char *o, double x, int up, int had_sign) {
  if (x != x) {
    const char *s = up ? "NAN" : "nan";
    for (int i = 0; i < 3; i++) o[had_sign + i] = s[i];
    return had_sign + 3;
  }
  if (x > 1.7976931348623157e308 || x < -1.7976931348623157e308) {
    const char *s = up ? "INF" : "inf";
    for (int i = 0; i < 3; i++) o[had_sign + i] = s[i];
    return had_sign + 3;
  }
  return -1; /* finite */
}

/* Round `x` (>= 0) to `prec` fractional digits and write "int.frac" (no sign). Returns bytes. */
static int svm_fmt_fixed_mag(char *o, double x, int prec) {
  int n = 0;
  /* Scale, round to integer, then split. Guard against overflow for large magnitudes by capping. */
  double scale = 1.0;
  for (int i = 0; i < prec; i++) scale *= 10.0;
  double scaled = x * scale + 0.5; /* round half up */
  /* integer part digits of `scaled` */
  char digs[512];
  int nd = 0;
  double ip = scaled;
  /* extract digits from the (possibly huge) magnitude via repeated /10 on a double; exact enough for
   * the plausible range, best-effort by contract. */
  if (ip < 1.0) {
    digs[nd++] = '0';
  } else {
    char rev[512];
    int r = 0;
    while (ip >= 1.0 && r < (int)sizeof rev) {
      double q = ip / 10.0;
      double qf = (double)(long long)q; /* floor for non-negative */
      int d = (int)(ip - qf * 10.0);
      if (d < 0) d = 0;
      if (d > 9) d = 9;
      rev[r++] = (char)('0' + d);
      ip = qf;
    }
    for (int i = 0; i < r; i++) digs[nd++] = rev[r - 1 - i];
  }
  /* `digs` holds all digits of round(x*10^prec); the last `prec` are fractional. */
  int int_len = nd - prec;
  if (int_len <= 0) {
    o[n++] = '0';
    if (prec > 0) {
      o[n++] = '.';
      for (int i = 0; i < -int_len; i++) o[n++] = '0';
      for (int i = 0; i < nd; i++) o[n++] = digs[i];
    }
  } else {
    for (int i = 0; i < int_len; i++) o[n++] = digs[i];
    if (prec > 0) {
      o[n++] = '.';
      for (int i = int_len; i < nd; i++) o[n++] = digs[i];
    }
  }
  return n;
}

int __vm_fmt_fix(char *o, double x, int prec, int width, int flags) {
  (void)width; /* the shim applies field width/justification */
  int n = svm_fmt_sign(o, &x, flags);
  int sp = svm_fmt_special(o, x, (flags >> 3) & 1, n);
  if (sp >= 0) return sp;
  return n + svm_fmt_fixed_mag(o + n, x, prec < 0 ? 6 : prec);
}

int __vm_fmt_sci(char *o, double x, int prec, int width, int flags) {
  (void)width;
  int n = svm_fmt_sign(o, &x, flags);
  int sp = svm_fmt_special(o, x, (flags >> 3) & 1, n);
  if (sp >= 0) return sp;
  int up = (flags >> 3) & 1;
  int exp = 0;
  if (x != 0.0) {
    while (x >= 10.0) {
      x /= 10.0;
      exp++;
    }
    while (x < 1.0) {
      x *= 10.0;
      exp--;
    }
  }
  n += svm_fmt_fixed_mag(o + n, x, prec < 0 ? 6 : prec);
  o[n++] = up ? 'E' : 'e';
  o[n++] = exp < 0 ? '-' : '+';
  if (exp < 0) exp = -exp;
  o[n++] = (char)('0' + (exp / 10) % 10);
  o[n++] = (char)('0' + exp % 10);
  return n;
}

int __vm_fmt_gen(char *o, double x, int prec, int width, int flags) {
  /* %g: pick %e when the exponent is < -4 or >= precision, else %f; precision is significant digits
   * (min 1). Best-effort — does not strip trailing zeros the way glibc's %g does. */
  double ax = x < 0.0 ? -x : x;
  int exp = 0;
  if (ax != 0.0 && ax == ax && ax <= 1.7976931348623157e308) {
    double t = ax;
    while (t >= 10.0) {
      t /= 10.0;
      exp++;
    }
    while (t < 1.0) {
      t *= 10.0;
      exp--;
    }
  }
  int p = prec < 0 ? 6 : (prec == 0 ? 1 : prec);
  if (exp < -4 || exp >= p) return __vm_fmt_sci(o, x, p - 1, width, flags);
  return __vm_fmt_fix(o, x, p - 1 - exp, width, flags);
}

#include "../postgres/printf_shim.c"
