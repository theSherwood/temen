#ifndef __STDIO_H
#define __STDIO_H

// A small, self-contained <stdio.h> for the browser playground's C compiler (SELFHOST_C.md §7).
// The program the user writes is compiled by chibicc-the-guest and run under the fixed powerbox,
// which provides ambient `write`/`read`/`exit` (§3e). Everything here is guest C compiled *into*
// the program — no libc is linked — so `printf` is just formatting over `write(1, …)`. Integer,
// char, string, and pointer conversions plus `%f`/`%e`/`%g` (guest-C float formatting, correctly
// rounded to the requested precision — not a shortest-round-trip bignum dtoa, so `%.17g` and a few
// exact-tie roundings can differ from glibc; see the float helpers below) are all supported.

#include <stdarg.h>

// Ambient powerbox syscalls (the frontend lowers these to `cap.call` on the stashed handles).
int write(int fd, char *buf, long n);
int read(int fd, char *buf, long n);
void exit(int code);

typedef unsigned long size_t;
typedef long ssize_t;

// A FILE* is just its fd (0/1/2). No buffered stdio object — the powerbox stream is the buffer.
typedef void FILE;
#define stdin ((FILE *)0)
#define stdout ((FILE *)1)
#define stderr ((FILE *)2)
#define EOF (-1)

// ---- output sink: either an fd (batched through a local buffer) or a caller string (s[n]printf) ----
struct __pf_sink {
  char *out;   // non-null → writing into a string (snprintf); null → writing to fd
  size_t cap;  // string capacity (incl. space for the NUL)
  size_t n;    // total chars the full output *would* be (printf return value)
  int fd;      // target fd when out == 0
  char buf[128];
  int bn;
};
static void __pf_flush(struct __pf_sink *s) {
  if (!s->out && s->bn) {
    write(s->fd, s->buf, s->bn);
    s->bn = 0;
  }
}
static void __pf_emit(struct __pf_sink *s, char c) {
  if (s->out) {
    if (s->n + 1 < s->cap)
      s->out[s->n] = c;
  } else {
    s->buf[s->bn++] = c;
    if (s->bn == (int)sizeof(s->buf))
      __pf_flush(s);
  }
  s->n++;
}

// Format `val` in `base` (10/16/8) into `tmp` (reversed), return its length. `upper` picks the
// hex digit case.
static int __pf_utoa(unsigned long val, unsigned base, int upper, char *tmp) {
  const char *lo = "0123456789abcdef";
  const char *hi = "0123456789ABCDEF";
  const char *dig = upper ? hi : lo;
  int n = 0;
  if (val == 0)
    tmp[n++] = '0';
  while (val) {
    tmp[n++] = dig[val % base];
    val /= base;
  }
  return n;
}

// ---- float formatting (%f/%e/%g) — guest C, no bignum ----------------------------------------
// A double is formatted **correctly rounded to the requested precision** (a trailing guard digit
// carries the rounding). This is not a shortest-round-trip dtoa: `%.17g` of an arbitrary double may
// print a different digit tail than glibc, and magnitudes past ~1e18 (the u64 integer-part range)
// lose precision. For the values a playground uses it matches glibc. The helpers below take a
// **non-negative** magnitude; sign/Inf/NaN/padding are handled at the call site.

static double __pf_pow10(int e) {
  double r = 1.0, b = 10.0;
  int neg = e < 0;
  if (neg) e = -e;
  while (e) {
    if (e & 1) r *= b;
    b *= b;
    e >>= 1;
  }
  return neg ? 1.0 / r : r;
}

// `x` (>= 0) → "[ip].[frac]" with `prec` fractional digits (rounded). Returns the length.
static int __pf_fix(char *out, double x, int prec) {
  if (prec < 0) prec = 6;
  if (prec > 30) prec = 30;
  unsigned long long ip = (unsigned long long)x; // integer part (exact for |x| < 2^64)
  double frac = x - (double)ip;
  char fd[40];
  for (int k = 0; k <= prec; k++) { // prec digits + one guard
    frac *= 10.0;
    int d = (int)frac;
    if (d > 9) d = 9;
    else if (d < 0) d = 0;
    fd[k] = (char)d;
    frac -= d;
  }
  // Round half up on the guard digit (carrying into `ip` if it overflows). Without arbitrary
  // precision we can't reproduce glibc's true round-half-to-even: `0.05 * 10` already rounds to
  // *exactly* 0.5 in `double`, so an exact tie is indistinguishable from "just above". Half-up
  // matches glibc on the common non-representable decimals (0.05→0.1, 0.005→0.01) and matches the
  // schoolbook expectation on the rare exact halves (0.5→1), where glibc's banker's rounding differs.
  if (fd[prec] >= 5) {
    int k = prec - 1;
    for (; k >= 0; k--) {
      if (++fd[k] <= 9) break;
      fd[k] = 0;
    }
    if (k < 0) ip++;
  }
  int n = 0;
  char ib[24];
  int in = 0;
  if (ip == 0) ib[in++] = '0';
  while (ip) { ib[in++] = (char)('0' + (int)(ip % 10)); ip /= 10; }
  while (in) out[n++] = ib[--in];
  if (prec > 0) {
    out[n++] = '.';
    for (int k = 0; k < prec; k++) out[n++] = (char)('0' + fd[k]);
  }
  return n;
}

// `x` (>= 0) → "d.ddde±dd" with `prec` mantissa-fraction digits. Returns the length.
static int __pf_sci(char *out, double x, int prec, int upper) {
  if (prec < 0) prec = 6;
  int exp = 0;
  if (x != 0.0) {
    while (x >= 10.0) { x /= 10.0; exp++; }
    while (x < 1.0) { x *= 10.0; exp--; }
  }
  char m[48];
  int mn = __pf_fix(m, x, prec);
  int dot = 0;
  while (dot < mn && m[dot] != '.') dot++;
  if (dot >= 2) { // rounding pushed 9.99… up to 10.0 — renormalize to 1.0…, exp+1
    exp++;
    mn = 0;
    m[mn++] = '1';
    if (prec > 0) {
      m[mn++] = '.';
      for (int k = 0; k < prec; k++) m[mn++] = '0';
    }
  }
  int n = 0;
  for (int k = 0; k < mn; k++) out[n++] = m[k];
  out[n++] = upper ? 'E' : 'e';
  out[n++] = exp < 0 ? '-' : '+';
  int ae = exp < 0 ? -exp : exp;
  char eb[8];
  int en = 0;
  do { eb[en++] = (char)('0' + ae % 10); ae /= 10; } while (ae);
  while (en < 2) eb[en++] = '0'; // exponent is at least two digits
  while (en) out[n++] = eb[--en];
  return n;
}

// `x` (>= 0) → `%g`: %e or %f by exponent, trailing zeros stripped unless `alt`. Returns the length.
static int __pf_gen(char *out, double x, int prec, int upper, int alt) {
  int P = prec < 0 ? 6 : (prec == 0 ? 1 : prec);
  int exp = 0;
  {
    double t = x;
    if (t != 0.0) {
      while (t >= 10.0) { t /= 10.0; exp++; }
      while (t < 1.0) { t *= 10.0; exp--; }
    }
  }
  int use_e = (exp < -4 || exp >= P);
  int n = use_e ? __pf_sci(out, x, P - 1, upper) : __pf_fix(out, x, P - 1 - exp);
  if (!alt) {
    int end = n;
    if (use_e) { end = 0; while (end < n && out[end] != 'e' && out[end] != 'E') end++; }
    int has_dot = 0;
    for (int k = 0; k < end; k++) if (out[k] == '.') has_dot = 1;
    if (has_dot) {
      int j = end;
      while (j > 0 && out[j - 1] == '0') j--;
      if (j > 0 && out[j - 1] == '.') j--;
      if (use_e && end < n) { // close the gap before the exponent
        int shift = end - j;
        for (int k = end; k < n; k++) out[k - shift] = out[k];
        n -= shift;
      } else {
        n = j;
      }
    }
  }
  return n;
}

// Format the magnitude `x` (>= 0) per conv (f/e/g, any case). Returns the length.
static int __pf_float(char *out, double x, int prec, char conv, int alt) {
  int up = (conv <= 'Z');
  char c = up ? (char)(conv + 32) : conv;
  if (c == 'f') return __pf_fix(out, x, prec);
  if (c == 'e') return __pf_sci(out, x, prec, up);
  return __pf_gen(out, x, prec, up, alt);
}

static int __pf_vprint(struct __pf_sink *s, const char *fmt, va_list ap) {
  for (int i = 0; fmt[i]; i++) {
    if (fmt[i] != '%') {
      __pf_emit(s, fmt[i]);
      continue;
    }
    i++;
    // flags
    int left = 0, zero = 0, plus = 0, space = 0, alt = 0;
    for (;; i++) {
      if (fmt[i] == '-') left = 1;
      else if (fmt[i] == '0') zero = 1;
      else if (fmt[i] == '+') plus = 1;
      else if (fmt[i] == ' ') space = 1;
      else if (fmt[i] == '#') alt = 1;
      else break;
    }
    // width
    int width = 0;
    while (fmt[i] >= '0' && fmt[i] <= '9') { width = width * 10 + (fmt[i] - '0'); i++; }
    // precision
    int prec = -1;
    if (fmt[i] == '.') {
      i++;
      prec = 0;
      while (fmt[i] >= '0' && fmt[i] <= '9') { prec = prec * 10 + (fmt[i] - '0'); i++; }
    }
    // length modifiers (accepted, and `l`/`ll`/`z` widen the fetch to 64-bit)
    int lng = 0;
    while (fmt[i] == 'l' || fmt[i] == 'h' || fmt[i] == 'z') { if (fmt[i] == 'l' || fmt[i] == 'z') lng++; i++; }

    char conv = fmt[i];
    char tmp[32];
    char pre[3];
    int npre = 0;
    int digits = 0, neg = 0;
    int is_num = 0;

    if (conv == 'd' || conv == 'i') {
      long v = lng ? va_arg(ap, long) : (long)va_arg(ap, int);
      unsigned long u = (v < 0) ? (neg = 1, -(unsigned long)v) : (unsigned long)v;
      digits = __pf_utoa(u, 10, 0, tmp);
      if (neg) pre[npre++] = '-';
      else if (plus) pre[npre++] = '+';
      else if (space) pre[npre++] = ' ';
      is_num = 1;
    } else if (conv == 'u') {
      unsigned long u = lng ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned int);
      digits = __pf_utoa(u, 10, 0, tmp);
      is_num = 1;
    } else if (conv == 'x' || conv == 'X') {
      unsigned long u = lng ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned int);
      digits = __pf_utoa(u, 16, conv == 'X', tmp);
      is_num = 1;
    } else if (conv == 'o') {
      unsigned long u = lng ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned int);
      digits = __pf_utoa(u, 8, 0, tmp);
      is_num = 1;
    } else if (conv == 'p') {
      unsigned long u = (unsigned long)va_arg(ap, void *);
      digits = __pf_utoa(u, 16, 0, tmp);
      pre[npre++] = '0';
      pre[npre++] = 'x';
      is_num = 1;
    } else if (conv == 'f' || conv == 'F' || conv == 'e' || conv == 'E' || conv == 'g' || conv == 'G') {
      double x = va_arg(ap, double);
      int up = (conv <= 'Z');
      char body[128];
      int bn = 0;
      char sign = 0;
      if (x != x) { // NaN (never negative-signed)
        const char *w = up ? "NAN" : "nan";
        for (int k = 0; w[k]; k++) body[bn++] = w[k];
      } else {
        if (x < 0.0) { sign = '-'; x = -x; }
        else if (plus) sign = '+';
        else if (space) sign = ' ';
        if (x != 0.0 && x * 0.5 == x) { // +Inf
          const char *w = up ? "INF" : "inf";
          for (int k = 0; w[k]; k++) body[bn++] = w[k];
        } else {
          bn = __pf_float(body, x, prec, conv, alt);
        }
      }
      int finite = !(body[0] == 'n' || body[0] == 'N' || body[0] == 'i' || body[0] == 'I');
      int pad = width - (sign ? 1 : 0) - bn;
      if (!left && !(zero && finite)) while (pad-- > 0) __pf_emit(s, ' ');
      if (sign) __pf_emit(s, sign);
      if (!left && zero && finite) while (pad-- > 0) __pf_emit(s, '0');
      for (int k = 0; k < bn; k++) __pf_emit(s, body[k]);
      if (left) while (pad-- > 0) __pf_emit(s, ' ');
      continue;
    } else if (conv == 'c') {
      char c = (char)va_arg(ap, int);
      int pad = width - 1;
      if (!left) while (pad-- > 0) __pf_emit(s, ' ');
      __pf_emit(s, c);
      if (left) while (pad-- > 0) __pf_emit(s, ' ');
      continue;
    } else if (conv == 's') {
      char *str = va_arg(ap, char *);
      if (!str) str = "(null)";
      int len = 0;
      while (str[len] && (prec < 0 || len < prec)) len++;
      int pad = width - len;
      if (!left) while (pad-- > 0) __pf_emit(s, ' ');
      for (int k = 0; k < len; k++) __pf_emit(s, str[k]);
      if (left) while (pad-- > 0) __pf_emit(s, ' ');
      continue;
    } else if (conv == '%') {
      __pf_emit(s, '%');
      continue;
    } else {
      // Unknown/unsupported (e.g. %f) — echo it literally so the output is at least legible.
      __pf_emit(s, '%');
      if (conv) __pf_emit(s, conv);
      continue;
    }

    if (is_num) {
      // Precision on integers = minimum digit count (zero-padded); it overrides the '0' flag.
      int zeros = 0;
      if (prec >= 0) { if (digits < prec) zeros = prec - digits; zero = 0; }
      int body = npre + zeros + digits;
      int pad = width - body;
      if (!left && !zero) while (pad-- > 0) __pf_emit(s, ' ');
      for (int k = 0; k < npre; k++) __pf_emit(s, pre[k]);
      if (!left && zero) while (pad-- > 0) __pf_emit(s, '0');
      while (zeros-- > 0) __pf_emit(s, '0');
      while (digits > 0) __pf_emit(s, tmp[--digits]);
      if (left) while (pad-- > 0) __pf_emit(s, ' ');
    }
  }
  return (int)s->n;
}

static int vfprintf(FILE *stream, const char *fmt, va_list ap) {
  struct __pf_sink s;
  s.out = 0; s.cap = 0; s.n = 0; s.fd = (int)(long)stream; s.bn = 0;
  int r = __pf_vprint(&s, fmt, ap);
  __pf_flush(&s);
  return r;
}
static int fprintf(FILE *stream, const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = vfprintf(stream, fmt, ap);
  va_end(ap);
  return r;
}
static int printf(const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = vfprintf(stdout, fmt, ap);
  va_end(ap);
  return r;
}
static int vsnprintf(char *str, size_t size, const char *fmt, va_list ap) {
  struct __pf_sink s;
  s.out = str; s.cap = size; s.n = 0; s.fd = 0; s.bn = 0;
  int r = __pf_vprint(&s, fmt, ap);
  if (size) s.out[s.n < size ? s.n : size - 1] = 0;
  return r;
}
static int snprintf(char *str, size_t size, const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = vsnprintf(str, size, fmt, ap);
  va_end(ap);
  return r;
}
static int sprintf(char *str, const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = vsnprintf(str, (size_t)1 << 30, fmt, ap);
  va_end(ap);
  return r;
}

static int fputc(int c, FILE *stream) {
  char b = (char)c;
  write((int)(long)stream, &b, 1);
  return c;
}
static int putc(int c, FILE *stream) { return fputc(c, stream); }
static int putchar(int c) { return fputc(c, stdout); }
static int fputs(const char *s, FILE *stream) {
  long n = 0;
  while (s[n]) n++;
  write((int)(long)stream, (char *)s, n);
  return 0;
}
static int puts(const char *s) {
  fputs(s, stdout);
  return putchar('\n');
}
static size_t fwrite(const void *ptr, size_t sz, size_t nm, FILE *stream) {
  long n = (long)(sz * nm);
  if (n) write((int)(long)stream, (char *)ptr, n);
  return nm;
}

static int getchar(void) {
  char c;
  return read(0, &c, 1) == 1 ? (unsigned char)c : EOF;
}
static char *fgets(char *s, int size, FILE *stream) {
  int fd = (int)(long)stream, i = 0;
  while (i < size - 1) {
    char c;
    if (read(fd, &c, 1) != 1) { if (i == 0) return 0; break; }
    s[i++] = c;
    if (c == '\n') break;
  }
  s[i] = 0;
  return s;
}

#endif
