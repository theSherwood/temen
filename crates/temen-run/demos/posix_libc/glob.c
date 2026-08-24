/* glob(3)/globfree as real guest libc — #800 (bash umbrella #794).
 *
 * POSIX pathname expansion over the personality's memfs: the pattern is split
 * on `/` and walked segment by segment — a magic segment (`*` `?` `[`)
 * enumerates its directory through opendir/readdir (ops 14/15) and filters with
 * slice 1's `fnmatch` (`FNM_PERIOD`, so `*` skips dotfiles — the shell rule); a
 * literal segment extends the prefix (existence resolves at the next opendir or
 * the final stat). Results sort by default (GLOB_NOSORT to keep directory
 * order), NULL-terminated in `gl_pathv` with `gl_offs` leading NULLs under
 * GLOB_DOOFFS, appended under GLOB_APPEND. GLOB_MARK stats each result and
 * marks directories with a trailing `/`; GLOB_NOCHECK returns the pattern
 * itself on no match. `errfunc` is accepted and ignored (bash passes NULL) and
 * a directory that fails to open is skipped — GLOB_ERR aborts instead
 * (GLOB_ABORTED). A relative pattern is walked relative to getcwd (op 9) but
 * returned in pattern form. Flag values follow glibc.
 *
 * Self-contained modulo slice 1: declares its own `__px_` externs and expects
 * `fnmatch.c` concatenated earlier in the translation unit.
 */

long __px_malloc(int cap, long size);
long __px_free(int cap, long ptr);
long __px_opendir(int cap, long path, long len);
long __px_readdir(int cap, long dir, long buf, long capn);
long __px_closedir(int cap, long dir);
long __px_stat(int cap, long path, long len, long statbuf);
long __px_getcwd(int cap, long buf, long size);

int fnmatch(char *pat, char *str, int flags);

#define GLOB_ERR 1
#define GLOB_MARK 2
#define GLOB_NOSORT 4
#define GLOB_DOOFFS 8
#define GLOB_NOCHECK 16
#define GLOB_APPEND 32
#define GLOB_NOESCAPE 64

#define GLOB_NOSPACE 1
#define GLOB_ABORTED 2
#define GLOB_NOMATCH 3

typedef struct {
  long gl_pathc;
  char **gl_pathv;
  long gl_offs;
} glob_t;

static long gl_slen_(char *s) {
  long n = 0;
  while (s[n]) n = n + 1;
  return n;
}

static char *gl_strdup2_(char *a, char *b) {
  long la = gl_slen_(a);
  long lb = gl_slen_(b);
  char *d = (char *)__px_malloc(0, la + lb + 1);
  if (!d) return 0;
  long i;
  for (i = 0; i < la; i = i + 1) d[i] = a[i];
  for (i = 0; i < lb; i = i + 1) d[la + i] = b[i];
  d[la + lb] = 0;
  return d;
}

static int gl_cmp_(char *a, char *b) {
  long i = 0;
  while (a[i] && a[i] == b[i]) i = i + 1;
  return (unsigned char)a[i] - (unsigned char)b[i];
}

/* A growable char* vector over malloc (no realloc op: grow by copy). */
typedef struct {
  char **v;
  long n;
  long cap;
} GlVec;

static int gl_push_(GlVec *g, char *s) {
  if (g->n >= g->cap) {
    long ncap = g->cap ? g->cap * 2 : 8;
    char **nv = (char **)__px_malloc(0, ncap * sizeof(char *));
    if (!nv) return -1;
    long i;
    for (i = 0; i < g->n; i = i + 1) nv[i] = g->v[i];
    if (g->v) __px_free(0, (long)g->v);
    g->v = nv;
    g->cap = ncap;
  }
  g->v[g->n] = s;
  g->n = g->n + 1;
  return 0;
}

static void gl_drop_(GlVec *g) {
  long i;
  for (i = 0; i < g->n; i = i + 1)
    if (g->v[i]) __px_free(0, (long)g->v[i]);
  if (g->v) __px_free(0, (long)g->v);
  g->v = 0;
  g->n = 0;
  g->cap = 0;
}

/* Does this segment carry an unquoted wildcard? (`\` quotes unless NOESCAPE.) */
static int gl_magic_(char *s, int flags) {
  long i = 0;
  while (s[i]) {
    if (s[i] == '\\' && !(flags & GLOB_NOESCAPE) && s[i + 1]) i = i + 2;
    else if (s[i] == '*' || s[i] == '?' || s[i] == '[') return 1;
    else i = i + 1;
  }
  return 0;
}

/* Strip the `\` quotes from a literal segment into a fresh string (the memfs
   holds unquoted names). */
static char *gl_unquote_(char *s, int flags) {
  char *d = (char *)__px_malloc(0, gl_slen_(s) + 1);
  if (!d) return 0;
  long i = 0;
  long j = 0;
  while (s[i]) {
    if (s[i] == '\\' && !(flags & GLOB_NOESCAPE) && s[i + 1]) i = i + 1;
    d[j] = s[i];
    j = j + 1;
    i = i + 1;
  }
  d[j] = 0;
  return d;
}

/* stat() the path: 1 = directory, 0 = file, -1 = absent. */
static int gl_kind_(char *path) {
  long st[2];
  if (__px_stat(0, (long)path, gl_slen_(path), (long)st) != 0) return -1;
  /* S_IFDIR = 0040000 */
  if ((st[0] & 0170000) == 0040000) return 1;
  return 0;
}

int glob(char *pattern, int flags, void *errfunc, glob_t *pglob) {
  (void)errfunc; /* accepted, ignored (bash passes NULL) */
  if (!*pattern) return GLOB_NOMATCH;
  /* The walk prefix (a real memfs path) and the result prefix (pattern form)
     stay in step; they differ only for a relative pattern, walked under cwd. */
  GlVec cur;
  cur.v = 0;
  cur.n = 0;
  cur.cap = 0;
  char cwd[256];
  char *root = "";
  int rel = *pattern != '/';
  if (rel) {
    if (__px_getcwd(0, (long)cwd, 256) <= 0) return GLOB_ABORTED;
    long cl = gl_slen_(cwd);
    while (cl > 0 && cwd[cl - 1] == '/') { cl = cl - 1; cwd[cl] = 0; }
    root = cwd;
  }
  if (gl_push_(&cur, gl_strdup2_(root, "")) != 0) return GLOB_NOSPACE;

  /* Split the pattern into segments in place-ish: scan `/`-separated pieces. */
  char *p = pattern;
  while (*p == '/') p = p + 1;
  int aborted = 0;
  int oom = 0;
  while (*p && cur.n) {
    char seg[256];
    long sl = 0;
    while (p[sl] && p[sl] != '/' && sl < 255) { seg[sl] = p[sl]; sl = sl + 1; }
    seg[sl] = 0;
    p = p + sl;
    while (*p == '/') p = p + 1;
    int last = *p == 0;

    GlVec next;
    next.v = 0;
    next.n = 0;
    next.cap = 0;
    long ci;
    for (ci = 0; ci < cur.n; ci = ci + 1) {
      char *pre = cur.v[ci];
      if (!gl_magic_(seg, flags)) {
        /* Literal segment: extend the prefix; a bogus middle path just fails
           the next opendir and drops out, the final segment must stat. */
        char *lit = gl_unquote_(seg, flags);
        char *full = 0;
        if (lit) {
          char *withsep = gl_strdup2_(pre, "/");
          if (withsep) {
            full = gl_strdup2_(withsep, lit);
            __px_free(0, (long)withsep);
          }
          __px_free(0, (long)lit);
        }
        if (!full) { oom = 1; break; }
        if (!last || gl_kind_(full) >= 0) {
          if (gl_push_(&next, full) != 0) { oom = 1; break; }
        } else {
          __px_free(0, (long)full);
        }
        continue;
      }
      /* Magic segment: enumerate the prefix directory and filter. */
      char *dirpath = *pre ? pre : "/";
      long d = __px_opendir(0, (long)dirpath, gl_slen_(dirpath));
      if (d < 0) {
        if (flags & GLOB_ERR) { aborted = 1; break; }
        continue;
      }
      char name[256];
      for (;;) {
        long r = __px_readdir(0, d, (long)name, 256);
        if (r <= 0) break;
        if (fnmatch(seg, name,
                    4 | ((flags & GLOB_NOESCAPE) ? 2 : 0)) != 0) /* FNM_PERIOD | FNM_NOESCAPE? */
          continue;
        char *withsep = gl_strdup2_(pre, "/");
        char *full = withsep ? gl_strdup2_(withsep, name) : 0;
        if (withsep) __px_free(0, (long)withsep);
        if (!full || gl_push_(&next, full) != 0) { oom = 1; break; }
      }
      __px_closedir(0, d);
      if (oom) break;
    }
    gl_drop_(&cur);
    cur = next;
    if (aborted || oom) {
      gl_drop_(&cur);
      return oom ? GLOB_NOSPACE : GLOB_ABORTED;
    }
  }

  /* Pattern-form results for a relative pattern: strip the cwd prefix plus
     its joining '/' (cwd "/" trims to "", so the strip is just the slash). */
  long strip = rel ? gl_slen_(root) + 1 : 0;

  /* Sort (insertion — result sets are shell-sized) unless GLOB_NOSORT. */
  if (!(flags & GLOB_NOSORT)) {
    long i;
    for (i = 1; i < cur.n; i = i + 1) {
      char *k = cur.v[i];
      long j = i;
      while (j > 0 && gl_cmp_(cur.v[j - 1], k) > 0) {
        cur.v[j] = cur.v[j - 1];
        j = j - 1;
      }
      cur.v[j] = k;
    }
  }

  if (cur.n == 0 && !(flags & GLOB_NOCHECK)) {
    if (!(flags & GLOB_APPEND)) {
      pglob->gl_pathc = 0;
      pglob->gl_pathv = 0;
    }
    return GLOB_NOMATCH;
  }

  /* Build the final NULL-terminated pathv: DOOFFS leading NULLs, any APPENDed
     prior results, then these (marked, cwd-stripped). */
  long offs = (flags & GLOB_DOOFFS) ? pglob->gl_offs : 0;
  long oldc = (flags & GLOB_APPEND) ? pglob->gl_pathc : 0;
  long newc = cur.n ? cur.n : 1; /* NOCHECK: the pattern itself */
  char **pv = (char **)__px_malloc(0, (offs + oldc + newc + 1) * sizeof(char *));
  if (!pv) {
    gl_drop_(&cur);
    return GLOB_NOSPACE;
  }
  long i;
  for (i = 0; i < offs; i = i + 1) pv[i] = 0;
  for (i = 0; i < oldc; i = i + 1) pv[offs + i] = pglob->gl_pathv[offs + i];
  if (cur.n == 0) {
    /* GLOB_NOCHECK: no expansion — the pattern, verbatim. */
    pv[offs + oldc] = gl_strdup2_(pattern, "");
    if (!pv[offs + oldc]) {
      __px_free(0, (long)pv);
      return GLOB_NOSPACE;
    }
  } else {
    for (i = 0; i < cur.n; i = i + 1) {
      char *path = cur.v[i] + strip;
      int mark = (flags & GLOB_MARK) && gl_kind_(cur.v[i]) == 1;
      pv[offs + oldc + i] = gl_strdup2_(path, mark ? "/" : "");
      if (!pv[offs + oldc + i]) {
        gl_drop_(&cur);
        __px_free(0, (long)pv);
        return GLOB_NOSPACE;
      }
    }
  }
  pv[offs + oldc + newc] = 0;
  gl_drop_(&cur);
  if ((flags & GLOB_APPEND) && pglob->gl_pathv) __px_free(0, (long)pglob->gl_pathv);
  pglob->gl_pathc = oldc + newc;
  pglob->gl_pathv = pv;
  if (!(flags & GLOB_DOOFFS)) pglob->gl_offs = 0;
  return 0;
}

void globfree(glob_t *pglob) {
  if (!pglob->gl_pathv) return;
  long i;
  for (i = 0; i < pglob->gl_offs + pglob->gl_pathc; i = i + 1)
    if (pglob->gl_pathv[i]) __px_free(0, (long)pglob->gl_pathv[i]);
  __px_free(0, (long)pglob->gl_pathv);
  pglob->gl_pathv = 0;
  pglob->gl_pathc = 0;
}
