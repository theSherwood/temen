#ifndef __MATH_H
#define __MATH_H

// <math.h> for the playground — guest C, no authority. The algebraic functions (fabs/floor/ceil/
// trunc/round/fmod/sqrt) are exact-to-the-model; the transcendentals (exp/log/sin/cos/pow/…) are
// range-reduced series — **demo-accurate**, not correctly-rounded libm (a few ULPs off on hard
// inputs), the same posture as the `<stdio.h>` float formatter. Plenty for compiling and running real
// C in the sandbox; a program that needs bit-exact libm is out of scope here.

#define M_PI     3.14159265358979323846
#define M_PI_2   1.57079632679489661923
#define M_PI_4   0.78539816339744830961
#define M_E      2.71828182845904523536
#define M_SQRT2  1.41421356237309504880
#define M_LN2    0.69314718055994530942
#define M_LN10   2.30258509299404568402

// +inf / nan built at runtime (no literal 1.0/0.0 — chibicc rejects the constant divide-by-zero).
static inline double __pg_inf(void) { double x = 1e300; return x * x; }        // 1e600 → +inf
static inline double __pg_nan(void) { double i = __pg_inf(); return i - i; }   // inf-inf → nan
#define HUGE_VAL (__pg_inf())
#define INFINITY (__pg_inf())
#define NAN      (__pg_nan())

static inline int isnan(double x) { return x != x; }
static inline int isinf(double x) { return x != 0 && x * 0.5 == x && x == x; }
static inline int isfinite(double x) { return !isnan(x) && !isinf(x); }
static inline int signbit(double x) { return x < 0 || (x == 0 && 1.0 / x < 0); }

static inline double fabs(double x) { return x < 0 ? -x : x; }
static inline float fabsf(float x) { return x < 0 ? -x : x; }
static inline double copysign(double x, double y) { return signbit(y) ? -fabs(x) : fabs(x); }
static inline double fmax(double a, double b) { return a > b ? a : b; }
static inline double fmin(double a, double b) { return a < b ? a : b; }

// Truncate toward zero via the 64-bit integer path (exact for |x| < 2^63; huge values pass through).
static inline double trunc(double x) {
  if (!isfinite(x) || fabs(x) >= 9.2233720368547758e18) return x;
  return (double)(long long)x;
}
static inline double floor(double x) { double t = trunc(x); return (t > x) ? t - 1 : t; }
static inline double ceil(double x)  { double t = trunc(x); return (t < x) ? t + 1 : t; }
static inline double round(double x) { return x < 0 ? ceil(x - 0.5) : floor(x + 0.5); }
static inline double fmod(double a, double b) {
  if (b == 0 || !isfinite(a) || !isfinite(b)) return __pg_nan();
  return a - trunc(a / b) * b;
}
static inline double modf(double x, double *ip) { double t = trunc(x); *ip = t; return x - t; }
static inline double ldexp(double x, int e) {
  double p = 1, b = e < 0 ? 0.5 : 2; int n = e < 0 ? -e : e;
  for (int i = 0; i < n; i++) p *= b;
  return x * p;
}
static inline double frexp(double x, int *e) {
  int n = 0;
  if (x == 0 || !isfinite(x)) { *e = 0; return x; }
  double m = fabs(x);
  while (m >= 1) { m *= 0.5; n++; }
  while (m < 0.5) { m *= 2; n--; }
  *e = n;
  return x < 0 ? -m : m;
}

static inline double sqrt(double x) {
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
static inline double hypot(double a, double b) { return sqrt(a * a + b * b); }
static inline double cbrt(double x) {
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
static inline double exp(double x) {
  if (isnan(x)) return x;
  if (x > 709) return __pg_inf();
  if (x < -745) return 0;
  double k = round(x / M_LN2);
  double r = x - k * M_LN2, term = 1, sum = 1;
  for (int i = 1; i < 20; i++) { term *= r / i; sum += term; }
  return ldexp(sum, (int)k);
}
// log via log(m·2^k) = k·ln2 + log(m), m ∈ [√½, √2), atanh series log(m) = 2·Σ t^(2n+1)/(2n+1).
static inline double log(double x) {
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
static inline double log2(double x)  { return log(x) / M_LN2; }
static inline double log10(double x) { return log(x) / M_LN10; }
static inline double pow(double b, double e) {
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
static inline double __pg_sin_core(double x) {
  double term = x, sum = x, x2 = x * x;
  for (int i = 1; i < 12; i++) { term *= -x2 / ((2 * i) * (2 * i + 1)); sum += term; }
  return sum;
}
static inline double __pg_reduce(double x) {
  double twopi = 2 * M_PI;
  x = fmod(x, twopi);
  if (x > M_PI) x -= twopi; else if (x < -M_PI) x += twopi;
  return x;
}
static inline double sin(double x) { return __pg_sin_core(__pg_reduce(x)); }
static inline double cos(double x) { return __pg_sin_core(__pg_reduce(x + M_PI_2)); }
static inline double tan(double x) { double c = cos(x); return c == 0 ? __pg_nan() : sin(x) / c; }
// atan via range-folded series; atan2 from atan with quadrant fixups; asin/acos from atan.
static inline double atan(double x) {
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
static inline double atan2(double y, double x) {
  if (x > 0) return atan(y / x);
  if (x < 0) return y >= 0 ? atan(y / x) + M_PI : atan(y / x) - M_PI;
  if (y > 0) return M_PI_2;
  if (y < 0) return -M_PI_2;
  return 0;
}
static inline double asin(double x) {
  if (x < -1 || x > 1) return __pg_nan();
  if (x == 1) return M_PI_2; if (x == -1) return -M_PI_2;
  return atan(x / sqrt(1 - x * x));
}
static inline double acos(double x) { return M_PI_2 - asin(x); }

#endif
