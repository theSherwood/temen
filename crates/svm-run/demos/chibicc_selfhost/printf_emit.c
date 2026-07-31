/* Emit-object printf (SELFHOST_C.md §7, task #20 increment 2b; float-input increment 2c) — the varargs
 * `vfprintf`/`fprintf`/`printf`/`snprintf`/`vsnprintf` family for the path where *chibicc itself*
 * compiles the libc.
 *
 * The on-ramp's `printf_shim.c` engine is already pure C for the whole integer/string/char/pointer +
 * width/precision/flags surface chibicc uses (`%d`/`%s`/`%x`/`%ld`/`%02d`/`%.*s`/`%+ld`/`%*s`); its
 * *only* on-ramp dependency is the float path, which delegates to the svm-llvm intrinsics
 * `__vm_fmt_{fix,sci,gen}` (a correctly-rounded bignum dtoa) that chibicc's `--emit-object` codegen does
 * not lower. So this file **defines those three as guest C** and then includes the shim verbatim — one
 * engine, all formats, no intrinsics. (The shim declares them `extern`; defining them first in the same
 * TU satisfies that.)
 *
 * The guest float path is **correctly rounded** (task #22): a Steele & White scaled-bignum digit
 * generator (a faithful C port of svm-llvm's `synth_dtoa_*` IR reference, `crates/svm-llvm/src/lib.rs`)
 * produces the nearest half-to-even P-significant-digit decimal using *exact* big-integer arithmetic —
 * no float ops in the digit loop — so `%.17g` yields the 17-significant-digit representation that
 * re-parses a `double` losslessly. chibicc emits float literals with `%.17g` (codegen_ir.c), so this is
 * what the emit-object self-compile of a float-bearing program needs. Validated byte-for-byte against
 * glibc across edge cases + random doubles (both signs; %g/%e/%f at many precisions). Field
 * width/justification and the `0`/`#` flags stay in the shim; here we emit the unpadded
 * `[sign]digits…` content only. */

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

/* ---- correctly-rounded significant-digit engine (Steele & White scaled bignum) ----
 * A fixed-width little-endian big integer (32-bit limbs). 72 × 32 = 2304 bits covers the full
 * double range: the largest scaling is the min subnormal (~10^-324), whose num/den need < 1200 bits. */
#define FBIG_LIMBS 72
typedef struct {
  unsigned int v[FBIG_LIMBS];
} fbig;

static void fbig_zero(fbig *a) {
  for (int i = 0; i < FBIG_LIMBS; i++) a->v[i] = 0;
}
static void fbig_copy(fbig *d, const fbig *s) {
  for (int i = 0; i < FBIG_LIMBS; i++) d->v[i] = s->v[i];
}
static int fbig_cmp(const fbig *a, const fbig *b) { /* -1 / 0 / +1 by magnitude */
  for (int i = FBIG_LIMBS - 1; i >= 0; i--)
    if (a->v[i] != b->v[i]) return a->v[i] < b->v[i] ? -1 : 1;
  return 0;
}
static void fbig_sub(fbig *a, const fbig *b) { /* a -= b, assumes a >= b */
  long long borrow = 0;
  for (int i = 0; i < FBIG_LIMBS; i++) {
    long long cur = (long long)a->v[i] - (long long)b->v[i] - borrow;
    if (cur < 0) {
      cur += 0x100000000LL;
      borrow = 1;
    } else {
      borrow = 0;
    }
    a->v[i] = (unsigned int)cur;
  }
}
static void fbig_mul_small(fbig *a, unsigned int m) { /* a *= m (m a small constant: 2, 5, 10) */
  unsigned long long carry = 0;
  for (int i = 0; i < FBIG_LIMBS; i++) {
    carry += (unsigned long long)a->v[i] * m;
    a->v[i] = (unsigned int)carry;
    carry >>= 32;
  }
}
static void fbig_shl(fbig *a, int n) { /* a <<= n bits (n >= 0) */
  if (n <= 0) return;
  int limbs = n >> 5, bits = n & 31;
  if (limbs)
    for (int i = FBIG_LIMBS - 1; i >= 0; i--) a->v[i] = (i - limbs >= 0) ? a->v[i - limbs] : 0;
  if (bits) {
    unsigned int carry = 0;
    for (int i = 0; i < FBIG_LIMBS; i++) {
      unsigned long long x = ((unsigned long long)a->v[i] << bits) | carry;
      a->v[i] = (unsigned int)x;
      carry = (unsigned int)(x >> 32);
    }
  }
}

/* Decode `x` (finite, > 0) into scaled bignums num/den and return E, the base-10 exponent of the
 * leading significant digit: after fixup num ∈ [den, 10·den), so value = (num/den) × 10^E with a
 * leading digit in [1,9]. No rounding happens here. */
static int dtoa_prep(double x, fbig *num, fbig *den) {
  union {
    double d;
    unsigned long long u;
  } pun;
  pun.d = x;
  unsigned long long bits = pun.u;
  int exp = (int)((bits >> 52) & 0x7FF);
  unsigned long long mant = bits & (((unsigned long long)1 << 52) - 1);
  unsigned long long f;
  int ebin;
  if (exp == 0) { /* subnormal */
    f = mant;
    ebin = -1074;
  } else {
    f = mant | ((unsigned long long)1 << 52);
    ebin = exp - 1075;
  }
  fbig_zero(num);
  fbig_zero(den);
  num->v[0] = (unsigned int)(f & 0xFFFFFFFF);
  num->v[1] = (unsigned int)(f >> 32);
  den->v[0] = 1;
  /* E estimate = floor(log10(2^(ebin+52))) ≈ ((ebin+52)·1233) >> 12; fixup below corrects it. */
  int E = ((ebin + 52) * 1233) >> 12;
  if (ebin < 0) fbig_shl(den, -ebin);
  else fbig_shl(num, ebin);
  if (E < 0)
    for (int i = 0; i < -E; i++) fbig_mul_small(num, 10);
  else
    for (int i = 0; i < E; i++) fbig_mul_small(den, 10);
  fbig tmp;
  for (;;) { /* while num >= 10·den: den *= 10, E++ */
    fbig_copy(&tmp, den);
    fbig_mul_small(&tmp, 10);
    if (fbig_cmp(num, &tmp) < 0) break;
    fbig_copy(den, &tmp);
    E++;
  }
  while (fbig_cmp(num, den) < 0) { /* while num < den: num *= 10, E-- */
    fbig_mul_small(num, 10);
    E--;
  }
  return E;
}

/* Generate `nsig` correctly-rounded (half-to-even) digits of num/den into digs[0..nsig-1] (values
 * 0..9). `*E` is bumped on a 999…→1000… carry-out. num/den are consumed. */
static void dtoa_generate(fbig *num, fbig *den, int nsig, unsigned char *digs, int *E) {
  for (int j = 0; j < nsig; j++) {
    int d = 0;
    while (fbig_cmp(num, den) >= 0) {
      fbig_sub(num, den);
      d++;
    }
    digs[j] = (unsigned char)d;
    if (j + 1 < nsig) fbig_mul_small(num, 10);
  }
  fbig_mul_small(num, 2); /* round: compare 2·remainder to den */
  int c = fbig_cmp(num, den);
  int up = c > 0 || (c == 0 && (digs[nsig - 1] & 1));
  if (up) {
    int k = nsig - 1;
    for (;;) {
      if (k < 0) { /* all-nines carry-out: "1" + zeros, exponent up one */
        digs[0] = 1;
        for (int i = 1; i < nsig; i++) digs[i] = 0;
        (*E)++;
        break;
      }
      if (++digs[k] < 10) break;
      digs[k] = 0;
      k--;
    }
  }
}

/* Fill digs[0..nsig-1] with the correctly-rounded significant digits of x (finite, > 0) and return
 * the leading digit's base-10 exponent E. nsig >= 1. */
static int dtoa_digits(double x, int nsig, unsigned char *digs) {
  fbig num, den;
  int E = dtoa_prep(x, &num, &den);
  dtoa_generate(&num, &den, nsig, digs, &E);
  return E;
}

/* Write "e±XX" (min two exponent digits, matching C printf). `up` selects 'E'. Returns bytes. */
static int svm_fmt_exp(char *o, int E, int up) {
  int n = 0;
  o[n++] = up ? 'E' : 'e';
  o[n++] = E < 0 ? '-' : '+';
  int ae = E < 0 ? -E : E;
  char t[8];
  int tn = 0;
  do {
    t[tn++] = (char)('0' + ae % 10);
    ae /= 10;
  } while (ae);
  while (tn < 2) t[tn++] = '0';
  for (int i = tn - 1; i >= 0; i--) o[n++] = t[i];
  return n;
}

int __vm_fmt_sci(char *o, double x, int prec, int width, int flags) {
  (void)width; /* the shim applies field width/justification */
  int n = svm_fmt_sign(o, &x, flags);
  int sp = svm_fmt_special(o, x, (flags >> 3) & 1, n);
  if (sp >= 0) return sp;
  int up = (flags >> 3) & 1;
  int fp = prec < 0 ? 6 : prec;
  if (fp > 480) fp = 480; /* keep content within the shim's 512-byte staging buffer */
  if (x == 0.0) {
    o[n++] = '0';
    if (fp > 0) {
      o[n++] = '.';
      for (int i = 0; i < fp; i++) o[n++] = '0';
    }
    return n + svm_fmt_exp(o + n, 0, up);
  }
  unsigned char digs[512];
  int E = dtoa_digits(x, fp + 1, digs); /* fp fractional digits ⇒ fp+1 significant */
  o[n++] = (char)('0' + digs[0]);
  if (fp > 0) {
    o[n++] = '.';
    for (int i = 1; i <= fp; i++) o[n++] = (char)('0' + digs[i]);
  }
  return n + svm_fmt_exp(o + n, E, up);
}

int __vm_fmt_fix(char *o, double x, int prec, int width, int flags) {
  (void)width;
  int n = svm_fmt_sign(o, &x, flags);
  int sp = svm_fmt_special(o, x, (flags >> 3) & 1, n);
  if (sp >= 0) return sp;
  int fp = prec < 0 ? 6 : prec;
  if (x == 0.0) {
    o[n++] = '0';
    if (fp > 0) {
      o[n++] = '.';
      for (int i = 0; i < fp; i++) o[n++] = '0';
    }
    return n;
  }
  fbig num, den;
  int E = dtoa_prep(x, &num, &den);
  int nsig = E + 1 + fp; /* significant digits down to the 10^-fp place */
  if (nsig <= 0) {
    /* |x| < 10^-fp: rounds to 0, or to a single 1 at the last place. Only nsig==0 can round up, and
       then only when num/(10·den) > 1/2 (half-to-even ⇒ an exact half rounds to the even 0). */
    int up = 0;
    if (nsig == 0) {
      fbig t;
      fbig_copy(&t, &den);
      fbig_mul_small(&t, 5);
      up = fbig_cmp(&num, &t) > 0;
    }
    if (fp == 0) {
      o[n++] = up ? '1' : '0';
      return n;
    }
    o[n++] = '0';
    o[n++] = '.';
    for (int i = 0; i < fp; i++) o[n++] = '0';
    if (up) o[n - 1] = '1';
    return n;
  }
  if (nsig > 480) nsig = 480;
  unsigned char digs[512];
  dtoa_generate(&num, &den, nsig, digs, &E);
  int intlen = E + 1;
  if (intlen <= 0)
    o[n++] = '0'; /* |x| < 1: the "0" before the point */
  else
    for (int i = 0; i < intlen; i++) o[n++] = (char)('0' + (i < nsig ? digs[i] : 0));
  if (fp > 0) {
    o[n++] = '.';
    for (int i = 0; i < fp; i++) {
      int idx = intlen + i;
      o[n++] = (char)('0' + (idx >= 0 && idx < nsig ? digs[idx] : 0));
    }
  }
  return n;
}

int __vm_fmt_gen(char *o, double x, int prec, int width, int flags) {
  /* %g: P significant digits (min 1), correctly rounded, trailing zeros stripped (no `#`); %e form
   * when the leading-digit exponent E < -4 or E >= P, else %f form. */
  (void)width;
  int n = svm_fmt_sign(o, &x, flags);
  int sp = svm_fmt_special(o, x, (flags >> 3) & 1, n);
  if (sp >= 0) return sp;
  int up = (flags >> 3) & 1;
  int P = prec < 0 ? 6 : (prec == 0 ? 1 : prec);
  if (P > 480) P = 480;
  if (x == 0.0) {
    o[n++] = '0';
    return n;
  }
  unsigned char digs[512];
  int E = dtoa_digits(x, P, digs);
  int last = P - 1; /* strip trailing zeros (keep at least the leading digit) */
  while (last > 0 && digs[last] == 0) last--;
  int ndig = last + 1;
  if (E < -4 || E >= P) {
    o[n++] = (char)('0' + digs[0]);
    if (ndig > 1) {
      o[n++] = '.';
      for (int i = 1; i < ndig; i++) o[n++] = (char)('0' + digs[i]);
    }
    n += svm_fmt_exp(o + n, E, up);
  } else if (E >= 0) {
    int intlen = E + 1;
    for (int i = 0; i < intlen; i++) o[n++] = (char)('0' + (i < ndig ? digs[i] : 0));
    if (ndig > intlen) {
      o[n++] = '.';
      for (int i = intlen; i < ndig; i++) o[n++] = (char)('0' + digs[i]);
    }
  } else { /* E in [-4,-1]: 0.<(-E-1) zeros><significant digits> */
    o[n++] = '0';
    o[n++] = '.';
    for (int i = 0; i < -E - 1; i++) o[n++] = '0';
    for (int i = 0; i < ndig; i++) o[n++] = (char)('0' + digs[i]);
  }
  return n;
}

#include "../postgres/printf_shim.c"
