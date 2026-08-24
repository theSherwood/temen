/* fnmatch(3) as real guest libc — #800 (bash umbrella #794).
 *
 * POSIX shell pattern matching with the flags bash uses: FNM_PATHNAME (globbing:
 * wildcards never cross `/`), FNM_PERIOD (a leading dot only matches an explicit
 * dot), FNM_NOESCAPE (`\` is a literal), and FNM_CASEFOLD (the GNU/BSD extension
 * behind nocaseglob/nocasematch). Bracket expressions carry ranges, negation
 * (`[!...]` and `[^...]`), a literal `]` in first position, and the named classes
 * (`[[:alpha:]]` …) bash's `case` relies on; an unterminated or unknown-class
 * bracket falls back to a literal `[` (the glibc behavior, which keeps shell
 * patterns like `[` usable). Matching is iterative with the classic single-star
 * backtrack point; under FNM_PATHNAME the backtrack never crosses a `/`, so the
 * worst case stays linear per segment.
 *
 * Pure compute, no personality ops — this file is self-contained (freestanding:
 * no includes) and concatenation-friendly for the c_posix test harness, which
 * differential-tests it against the host's fnmatch(3). Flag values follow
 * glibc/musl; the harness maps them to the host's constants per platform.
 */

#define FNM_NOMATCH 1

#define FNM_PATHNAME 1
#define FNM_NOESCAPE 2
#define FNM_PERIOD 4
#define FNM_CASEFOLD 16

static int fnm_lower_(int c) {
  if (c >= 'A' && c <= 'Z') return c - 'A' + 'a';
  return c;
}

static int fnm_eq_(int a, int b, int flags) {
  if (a == b) return 1;
  if (flags & FNM_CASEFOLD) return fnm_lower_(a) == fnm_lower_(b);
  return 0;
}

static int fnm_is_alpha_(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static int fnm_is_digit_(int c) { return c >= '0' && c <= '9'; }
static int fnm_is_space_(int c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}
static int fnm_is_punct_(int c) {
  return c > ' ' && c < 127 && !fnm_is_alpha_(c) && !fnm_is_digit_(c);
}

static int fnm_streq_(char *a, char *b) {
  while (*a && *a == *b) { a = a + 1; b = b + 1; }
  return *a == *b;
}

/* Match `c` against the named class at `p` (just past "[:"), leaving `*end` past the
   closing ":]" on success. Returns -1 for an unknown or unterminated class name. */
static int fnm_class_(char *p, int c, char **end) {
  char name[8];
  int n = 0;
  while (p[n] && p[n] != ':' && n < 7) { name[n] = p[n]; n = n + 1; }
  if (p[n] != ':' || p[n + 1] != ']') return -1;
  name[n] = 0;
  *end = p + n + 2;
  if (fnm_streq_(name, "alpha")) return fnm_is_alpha_(c);
  if (fnm_streq_(name, "digit")) return fnm_is_digit_(c);
  if (fnm_streq_(name, "alnum")) return fnm_is_alpha_(c) || fnm_is_digit_(c);
  if (fnm_streq_(name, "upper")) return c >= 'A' && c <= 'Z';
  if (fnm_streq_(name, "lower")) return c >= 'a' && c <= 'z';
  if (fnm_streq_(name, "space")) return fnm_is_space_(c);
  if (fnm_streq_(name, "blank")) return c == ' ' || c == '\t';
  if (fnm_streq_(name, "punct")) return fnm_is_punct_(c);
  if (fnm_streq_(name, "print")) return c >= ' ' && c < 127;
  if (fnm_streq_(name, "graph")) return c > ' ' && c < 127;
  if (fnm_streq_(name, "cntrl")) return c < ' ' || c == 127;
  if (fnm_streq_(name, "xdigit"))
    return fnm_is_digit_(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
  return -1;
}

/* Match one bracket expression starting at `*pp` (just past the `[`) against `c`.
   On success or clean no-match, advance `*pp` past the closing `]` and return 1/0.
   Returns -1 when the expression is malformed (unterminated, bad class) — the
   caller then treats the `[` as a literal (glibc's fallback). */
static int fnm_bracket_(char **pp, int c, int flags) {
  char *p = *pp;
  int neg = 0;
  int hit = 0;
  if (*p == '!' || *p == '^') { neg = 1; p = p + 1; }
  int first = 1;
  while (*p && (*p != ']' || first)) {
    first = 0;
    if (p[0] == '[' && p[1] == ':') {
      char *end;
      int r = fnm_class_(p + 2, c, &end);
      if (r < 0) return -1;
      if (r) hit = 1;
      p = end;
      continue;
    }
    int lo = *p;
    if (lo == '\\' && !(flags & FNM_NOESCAPE) && p[1]) { p = p + 1; lo = *p; }
    p = p + 1;
    if (*p == '-' && p[1] && p[1] != ']') {
      int hi = p[1];
      p = p + 2;
      if (hi == '\\' && !(flags & FNM_NOESCAPE) && *p) { hi = *p; p = p + 1; }
      if ((c >= lo && c <= hi) ||
          ((flags & FNM_CASEFOLD) && fnm_lower_(c) >= fnm_lower_(lo) &&
           fnm_lower_(c) <= fnm_lower_(hi)))
        hit = 1;
    } else if (fnm_eq_(lo, c, flags)) {
      hit = 1;
    }
  }
  if (!*p) return -1;
  *pp = p + 1;
  if (neg) return !hit;
  return hit;
}

int fnmatch(char *pat, char *str, int flags) {
  char *p = pat;
  char *s = str;
  /* The single-star backtrack point: on a mismatch past a `*`, resume just after
     the star with the next candidate tail. `0` means no star is live. */
  char *bt_p = 0;
  char *bt_s = 0;
  /* FNM_PERIOD: a leading `.` (of the string, or of each segment under
     FNM_PATHNAME) matches only an explicit `.` in the pattern. */
  int at_start = 1;
  for (;;) {
    int pc = *p;
    if (!*s) {
      /* String exhausted: only trailing stars may remain. */
      while (*p == '*') p = p + 1;
      if (!*p) return 0;
      return FNM_NOMATCH;
    }
    if (pc == '*') {
      if ((flags & FNM_PERIOD) && at_start && *s == '.') return FNM_NOMATCH;
      p = p + 1;
      bt_p = p;
      bt_s = s;
      at_start = 0;
      continue;
    }
    int matched = 0;
    if (pc == '?') {
      matched = *s != 0;
      if ((flags & FNM_PATHNAME) && *s == '/') matched = 0;
      if ((flags & FNM_PERIOD) && at_start && *s == '.') matched = 0;
      if (matched) { p = p + 1; s = s + 1; at_start = 0; }
    } else if (pc == '[') {
      if (((flags & FNM_PATHNAME) && *s == '/') ||
          ((flags & FNM_PERIOD) && at_start && *s == '.')) {
        matched = 0;
      } else {
        char *q = p + 1;
        int r = fnm_bracket_(&q, *s, flags);
        if (r < 0) {
          /* Malformed: literal `[`. */
          matched = fnm_eq_('[', *s, flags);
          if (matched) { p = p + 1; s = s + 1; at_start = 0; }
        } else {
          matched = r;
          if (matched) { p = q; s = s + 1; at_start = 0; }
        }
      }
    } else {
      if (pc == '\\' && !(flags & FNM_NOESCAPE) && p[1]) { p = p + 1; pc = *p; }
      matched = fnm_eq_(pc, *s, flags);
      if (matched) {
        at_start = (flags & FNM_PATHNAME) && *s == '/';
        p = p + 1;
        s = s + 1;
      }
    }
    if (matched) continue;
    /* Mismatch: backtrack to the live star, advancing the string by one — but a
       star never swallows `/` under FNM_PATHNAME. */
    if (bt_p && *bt_s && !((flags & FNM_PATHNAME) && *bt_s == '/')) {
      bt_s = bt_s + 1;
      p = bt_p;
      s = bt_s;
      at_start = 0;
      continue;
    }
    return FNM_NOMATCH;
  }
}
