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

#include <__pg_linkage.h>

// ---- prototypes (a program unit, #1392) --------------------------------------------------
// Same split as <stdio.h>: a decls-only translation unit (`-include __pg_decls_only.h`) sees these
// prototypes and links against the prebuilt libc unit, instead of recompiling the series-based libm
// — the transcendentals are the bulk of this header — into every program. The bodies are compiled
// in by default, where `__PG_FN` makes them `static inline` so an unused one is dead-stripped —
// hence the guard, which would otherwise make every one of them a root.
#ifdef __PG_LIBC_DECLS_ONLY
double __pg_inf(void);
double __pg_nan(void);
int isnan(double x);
int isinf(double x);
int isfinite(double x);
int signbit(double x);
double fabs(double x);
float fabsf(float x);
double copysign(double x, double y);
double fmax(double a, double b);
double fmin(double a, double b);
double trunc(double x);
double floor(double x);
double ceil(double x);
double round(double x);
double fmod(double a, double b);
double modf(double x, double *ip);
double ldexp(double x, int e);
double frexp(double x, int *e);
double sqrt(double x);
double hypot(double a, double b);
double cbrt(double x);
double exp(double x);
double log(double x);
double log2(double x);
double log10(double x);
double pow(double b, double e);
double __pg_sin_core(double x);
double __pg_reduce(double x);
double sin(double x);
double cos(double x);
double tan(double x);
double atan(double x);
double atan2(double y, double x);
double asin(double x);
double acos(double x);
#endif /* __PG_LIBC_DECLS_ONLY */

// ---- bodies -----------------------------------------------------------------------------
// In their own file, not behind an `#ifdef` here: chibicc tokenizes a header in full before the
// preprocessor drops the skipped groups, so text left in place would still be *tokenized* by a
// decls-only compile. A separate file is never opened at all.
#ifndef __PG_LIBC_DECLS_ONLY
#include <__pg_math_impl.h>
#endif

#endif
