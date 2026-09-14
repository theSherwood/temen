#ifndef __PG_MATH_IMPL_H
#define __PG_MATH_IMPL_H

// The bodies of the seeded <math.h> (#1392) — see that header for why they live in their own
// file, and `__pg_linkage.h` for what `__PG_FN` means in each of the three compile modes.

#include <math.h>

__PG_FN double __pg_inf(void) { double x = 1e300; return x * x; }        // 1e600 → +inf
__PG_FN double __pg_nan(void) { double i = __pg_inf(); return i - i; }   // inf-inf → nan
#define HUGE_VAL (__pg_inf())
#define INFINITY (__pg_inf())
#define NAN      (__pg_nan())

__PG_FN int isnan(double x) { return x != x; }
__PG_FN int isinf(double x) { return x != 0 && x * 0.5 == x && x == x; }
__PG_FN int isfinite(double x) { return !isnan(x) && !isinf(x); }
__PG_FN int signbit(double x) { return x < 0 || (x == 0 && 1.0 / x < 0); }

__PG_FN double fabs(double x) { return x < 0 ? -x : x; }
__PG_FN float fabsf(float x) { return x < 0 ? -x : x; }
__PG_FN double copysign(double x, double y) { return signbit(y) ? -fabs(x) : fabs(x); }
__PG_FN double fmax(double a, double b) { return a > b ? a : b; }
__PG_FN double fmin(double a, double b) { return a < b ? a : b; }

// Truncate toward zero via the 64-bit integer path (exact for |x| < 2^63; huge values pass through).
__PG_FN double trunc(double x) {
  if (!isfinite(x) || fabs(x) >= 9.2233720368547758e18) return x;
  return (double)(long long)x;
}
__PG_FN double floor(double x) { double t = trunc(x); return (t > x) ? t - 1 : t; }
__PG_FN double ceil(double x)  { double t = trunc(x); return (t < x) ? t + 1 : t; }
__PG_FN double round(double x) { return x < 0 ? ceil(x - 0.5) : floor(x + 0.5); }
__PG_FN double fmod(double a, double b) {
  if (b == 0 || !isfinite(a) || !isfinite(b)) return __pg_nan();
  return a - trunc(a / b) * b;
}
__PG_FN double modf(double x, double *ip) { double t = trunc(x); *ip = t; return x - t; }
__PG_FN double ldexp(double x, int e) {
  double p = 1, b = e < 0 ? 0.5 : 2; int n = e < 0 ? -e : e;
  for (int i = 0; i < n; i++) p *= b;
  return x * p;
}
__PG_FN double frexp(double x, int *e) {
  int n = 0;
  if (x == 0 || !isfinite(x)) { *e = 0; return x; }
  double m = fabs(x);
  while (m >= 1) { m *= 0.5; n++; }
  while (m < 0.5) { m *= 2; n--; }
  *e = n;
  return x < 0 ? -m : m;
}

__PG_FN double sqrt(double x) {
  if (x < 0) return __pg_nan();
  if (x == 0 || !isfinite(x)) return x;
  double g = x > 1 ? x : 1;            // Newton's method: g ← (g + x/g)/2 to convergence
  for (int i = 0; i < 60; i++) {
    double ng = 0.5 * (g + x / g);
    if (ng == g) break;
    g = ng;
  }
  return g;
}
__PG_FN double hypot(double a, double b) { return sqrt(a * a + b * b); }
__PG_FN double cbrt(double x) {
  if (x == 0 || !isfinite(x)) return x;
  double s = x < 0 ? -1 : 1, y = fabs(x), g = y > 1 ? y : 1;
  for (int i = 0; i < 60; i++) {
    double ng = (2 * g + y / (g * g)) / 3;
    if (ng == g) break;
    g = ng;
  }
  return s * g;
}

// exp via 2^k·exp(r): k = round(x/ln2), r = x − k·ln2 ∈ [−ln2/2, ln2/2], Taylor on r.
__PG_FN double exp(double x) {
  if (isnan(x)) return x;
  if (x > 709) return __pg_inf();
  if (x < -745) return 0;
  double k = round(x / M_LN2);
  double r = x - k * M_LN2, term = 1, sum = 1;
  for (int i = 1; i < 20; i++) { term *= r / i; sum += term; }
  return ldexp(sum, (int)k);
}
// log via log(m·2^k) = k·ln2 + log(m), m ∈ [√½, √2), atanh series log(m) = 2·Σ t^(2n+1)/(2n+1).
__PG_FN double log(double x) {
  if (x < 0) return __pg_nan();
  if (x == 0) return -__pg_inf();
  if (!isfinite(x)) return x;
  int k = 0;
  while (x >= 1.41421356237309504880) { x *= 0.5; k++; }
  while (x < 0.70710678118654752440) { x *= 2; k--; }
  double t = (x - 1) / (x + 1), t2 = t * t, term = t, sum = 0;
  for (int i = 0; i < 30; i++) { sum += term / (2 * i + 1); term *= t2; }
  return 2 * sum + k * M_LN2;
}
__PG_FN double log2(double x)  { return log(x) / M_LN2; }
__PG_FN double log10(double x) { return log(x) / M_LN10; }
__PG_FN double pow(double b, double e) {
  if (e == 0) return 1;
  if (b == 0) return e > 0 ? 0 : __pg_inf();
  // Exact integer-exponent fast path (also the only correct route for b < 0).
  if (e == trunc(e) && fabs(e) < 1024) {
    long long n = (long long)fabs(e); double r = 1, p = b;
    while (n) { if (n & 1) r *= p; p *= p; n >>= 1; }
    return e < 0 ? 1 / r : r;
  }
  if (b < 0) return __pg_nan();
  return exp(e * log(b));
}

// sin/cos via reduction mod 2π then Taylor; tan = sin/cos.
__PG_FN double __pg_sin_core(double x) {
  double term = x, sum = x, x2 = x * x;
  for (int i = 1; i < 12; i++) { term *= -x2 / ((2 * i) * (2 * i + 1)); sum += term; }
  return sum;
}
__PG_FN double __pg_reduce(double x) {
  double twopi = 2 * M_PI;
  x = fmod(x, twopi);
  if (x > M_PI) x -= twopi; else if (x < -M_PI) x += twopi;
  return x;
}
__PG_FN double sin(double x) { return __pg_sin_core(__pg_reduce(x)); }
__PG_FN double cos(double x) { return __pg_sin_core(__pg_reduce(x + M_PI_2)); }
__PG_FN double tan(double x) { double c = cos(x); return c == 0 ? __pg_nan() : sin(x) / c; }
// atan via range-folded series; atan2 from atan with quadrant fixups; asin/acos from atan.
__PG_FN double atan(double x) {
  int neg = x < 0; if (neg) x = -x;
  int inv = x > 1; if (inv) x = 1 / x;
  int shift = x > 0.41421356237309504880; // tan(π/8): fold [√2−1, 1] down to keep the series short
  double c = 0;
  if (shift) { c = M_PI_4; x = (x - 1) / (x + 1); }
  double x2 = x * x, term = x, sum = x;
  for (int i = 1; i < 30; i++) { term *= -x2; sum += term / (2 * i + 1); }
  double r = c + sum;
  if (inv) r = M_PI_2 - r;
  return neg ? -r : r;
}
__PG_FN double atan2(double y, double x) {
  if (x > 0) return atan(y / x);
  if (x < 0) return y >= 0 ? atan(y / x) + M_PI : atan(y / x) - M_PI;
  if (y > 0) return M_PI_2;
  if (y < 0) return -M_PI_2;
  return 0;
}
__PG_FN double asin(double x) {
  if (x < -1 || x > 1) return __pg_nan();
  if (x == 1) return M_PI_2; if (x == -1) return -M_PI_2;
  return atan(x / sqrt(1 - x * x));
}
__PG_FN double acos(double x) { return M_PI_2 - asin(x); }


// ---- hyperbolics -------------------------------------------------------------------------
// Built on `exp`/`log`/`sqrt` above. nim's `std/math` imports these directly (`sinh`…`atanh`),
// so a nim program that pulls in `math` needs them to link (#1422).
__PG_FN double sinh(double x) { double e = exp(x); return (e - 1.0 / e) * 0.5; }
__PG_FN double cosh(double x) { double e = exp(x); return (e + 1.0 / e) * 0.5; }
__PG_FN double tanh(double x) {
  if (x > 20.0) return 1.0;                 // e^40 already saturates the ratio to 1
  if (x < -20.0) return -1.0;
  double e = exp(2.0 * x);
  return (e - 1.0) / (e + 1.0);
}
// `asinh` is odd; reflect negatives so the `x + sqrt(x^2+1)` never cancels catastrophically.
__PG_FN double asinh(double x) {
  return x < 0.0 ? -log(-x + sqrt(x * x + 1.0)) : log(x + sqrt(x * x + 1.0));
}
__PG_FN double acosh(double x) { return x < 1.0 ? __pg_nan() : log(x + sqrt(x * x - 1.0)); }
__PG_FN double atanh(double x) {
  if (x > 1.0 || x < -1.0) return __pg_nan();
  if (x == 1.0) return __pg_inf();
  if (x == -1.0) return -__pg_inf();
  return 0.5 * log((1.0 + x) / (1.0 - x));
}

// ---- float32 overloads -------------------------------------------------------------------
// nim declares every one of these beside its `float64` twin (`func sin*(x: float32): float32
// {.importc: "sinf".}`), so both spellings are imported whenever `std/math` is pulled in. Computing
// in `double` and narrowing is correct to `float` precision.
__PG_FN float sinf(float x) { return (float)sin((double)x); }
__PG_FN float cosf(float x) { return (float)cos((double)x); }
__PG_FN float tanf(float x) { return (float)tan((double)x); }
__PG_FN float asinf(float x) { return (float)asin((double)x); }
__PG_FN float acosf(float x) { return (float)acos((double)x); }
__PG_FN float atanf(float x) { return (float)atan((double)x); }
__PG_FN float sinhf(float x) { return (float)sinh((double)x); }
__PG_FN float coshf(float x) { return (float)cosh((double)x); }
__PG_FN float tanhf(float x) { return (float)tanh((double)x); }
__PG_FN float asinhf(float x) { return (float)asinh((double)x); }
__PG_FN float acoshf(float x) { return (float)acosh((double)x); }
__PG_FN float atanhf(float x) { return (float)atanh((double)x); }

#endif
