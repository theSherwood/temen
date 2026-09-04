/* cut(1) — select portions of each stdin line: fields (`-f LIST`, delimiter `-d C`, default TAB)
 * or characters (`-c LIST`). LIST is comma-separated 1-based selectors: `N`, `N-M`, `N-` (to end of
 * line), `-M` (from 1). In field mode a line with NO delimiter is passed through unchanged (GNU cut's
 * default, i.e. no `-s`); selected fields are re-joined with the delimiter, in ascending field order.
 * Reads line-at-a-time via `u_rdline` — a pipeline-polite citizen like the other #801 coreutils. */
long write(long fd, void *buf, long n);
long u_rdline(long fd, char *out, long cap);
long u_strlen(char *s);

/* A fixed selection bitmap (positions 1..MAXSEL) plus an open-ended `N-` low bound. Positions past
 * MAXSEL are only reachable through the open range, which is enough for the real pipeline uses. */
static char g_sel[4096];
static long g_open_from; /* 0 = no open range; else positions >= g_open_from are selected */

/* Parse a LIST into g_sel[]/g_open_from. Returns 0 on success, 1 on a malformed list. */
static int parse_list(char *s) {
  long i = 0;
  while (s[i]) {
    long a = 0, b = 0, hasa = 0, hasb = 0, dash = 0, p = 0;
    while (s[i] >= '0' && s[i] <= '9') { a = a * 10 + (s[i] - '0'); hasa = 1; i = i + 1; }
    if (s[i] == '-') {
      dash = 1; i = i + 1;
      while (s[i] >= '0' && s[i] <= '9') { b = b * 10 + (s[i] - '0'); hasb = 1; i = i + 1; }
    }
    if (s[i] == ',') i = i + 1;
    else if (s[i]) return 1; /* trailing junk */
    if (!dash) {
      if (!hasa || a < 1) return 1;
      if (a <= 4096) g_sel[a - 1] = 1;
    } else {
      long lo = hasa ? a : 1;
      if (lo < 1) return 1;
      if (hasb) {
        long hi = b;
        if (hi < lo) continue; /* empty range: skip */
        if (hi > 4096) hi = 4096;
        for (p = lo; p <= hi; p = p + 1) if (p >= 1 && p <= 4096) g_sel[p - 1] = 1;
      } else {
        if (g_open_from == 0 || lo < g_open_from) g_open_from = lo; /* `N-` to end */
      }
    }
  }
  return 0;
}

static int is_sel(long pos1) { /* pos1 is 1-based */
  if (pos1 >= 1 && pos1 <= 4096 && g_sel[pos1 - 1]) return 1;
  if (g_open_from > 0 && pos1 >= g_open_from) return 1;
  return 0;
}

int main(int argc, char **argv) {
  char delim = '\t';
  int mode = 0; /* 'f' fields, 'c' chars */
  char *list = 0;
  char nl = '\n';
  int i = 1;
  while (i < argc && argv[i][0] == '-' && argv[i][1]) {
    char *a = argv[i];
    if (a[1] == 'd') {
      if (a[2]) delim = a[2];
      else if (i + 1 < argc) { i = i + 1; delim = argv[i][0]; }
    } else if (a[1] == 'f') {
      mode = 'f';
      if (a[2]) list = a + 2;
      else if (i + 1 < argc) { i = i + 1; list = argv[i]; }
    } else if (a[1] == 'c') {
      mode = 'c';
      if (a[2]) list = a + 2;
      else if (i + 1 < argc) { i = i + 1; list = argv[i]; }
    } else {
      char *u = "cut: unknown option\n";
      write(2, u, u_strlen(u));
      return 2;
    }
    i = i + 1;
  }
  if (!mode || !list) {
    char *u = "cut: usage: cut -f LIST [-d C] | -c LIST\n";
    write(2, u, u_strlen(u));
    return 2;
  }
  if (parse_list(list)) {
    char *u = "cut: invalid field/character list\n";
    write(2, u, u_strlen(u));
    return 2;
  }

  char line[8192];
  long n, p, field, start, first, has;
  while ((n = u_rdline(0, line, 8192)) >= 0) {
    if (mode == 'c') {
      for (p = 0; p < n; p = p + 1) if (is_sel(p + 1)) write(1, line + p, 1);
      write(1, &nl, 1);
      continue;
    }
    /* field mode: a line with no delimiter passes through unchanged (GNU default) */
    has = 0;
    for (p = 0; p < n; p = p + 1) if (line[p] == delim) { has = 1; break; }
    if (!has) { write(1, line, n); write(1, &nl, 1); continue; }
    field = 1; start = 0; first = 1;
    for (p = 0; p <= n; p = p + 1) {
      if (p == n || line[p] == delim) {
        if (is_sel(field)) {
          if (!first) write(1, &delim, 1);
          write(1, line + start, p - start);
          first = 0;
        }
        field = field + 1;
        start = p + 1;
      }
    }
    write(1, &nl, 1);
  }
  return 0;
}
