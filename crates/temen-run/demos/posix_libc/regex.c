/* regcomp/regexec/regfree as real guest libc — #800 (bash umbrella #794).
 *
 * POSIX **ERE** (the only grammar bash's `[[ =~ ]]` uses — `regcomp` without
 * REG_EXTENDED is refused as REG_BADPAT): literals, `.`, `^`/`$`, bracket
 * expressions (ranges, `[^...]` negation, literal-`]`-first, the `[[:class:]]`
 * names), grouping with **captures** (`BASH_REMATCH`), alternation `|`, and the
 * duplications `*` `+` `?` `{n,m}`/`{n,}`/`{n}`. Flags: REG_ICASE (folded at
 * compile into chars and class bitmaps), REG_NOSUB, and REG_NOTBOL/REG_NOTEOL
 * at exec. REG_NEWLINE is not implemented (bash's `=~` never passes it).
 *
 * Shape: a recursive-descent parser builds an AST in a bump arena, a two-pass
 * emitter (exact `rx_size_` then `rx_emit_`) lowers it to a compact program of
 * relative-jump instructions — relative so `{n,m}` expands by block-copying the
 * body — and the matcher explores the program **exhaustively**, recording the
 * longest match at the leftmost start (POSIX leftmost-longest, where a
 * first-match backtracker would diverge on `(a|ab)`); at equal length the
 * greedy-first exploration order picks the capture set. A step budget bounds
 * pathological patterns (`(a*)*`): on exhaustion the best match found so far
 * stands, which for bash-sized patterns is never reached. Self-contained
 * (freestanding; declares its own `__px_` externs) for the c_posix harness,
 * which differential-tests it — spans and captures — against the host's
 * regexec(3).
 */

long __px_malloc(int cap, long size);
long __px_free(int cap, long ptr);

#define REG_EXTENDED 1
#define REG_ICASE 2
#define REG_NEWLINE 4
#define REG_NOSUB 8
#define REG_NOTBOL 1
#define REG_NOTEOL 2

#define REG_NOMATCH 1
#define REG_BADPAT 2
#define REG_ESPACE 12

#define RX_MAXSUB 32
#define RX_STEPS 200000

/* ABI note (#802 language differential): bash's TUs allocate `regex_t`/`regmatch_t` from the
 * build host's glibc <regex.h> and read `re_nsub` + the `pmatch` offsets across the call, so this
 * guest lib must match glibc's ABI byte-for-byte, NOT define a convenient layout. On the target
 * (x86-64 glibc): sizeof(regex_t)==64 with re_nsub@48, and regmatch_t is {int rm_so; int rm_eo}
 * (regoff_t is `int`, 8 bytes total). Getting this wrong still MATCHES (the internal fields are
 * self-consistent within the guest) but leaves BASH_REMATCH empty — bash read re_nsub and the
 * match offsets from the wrong places. The guest's own scratch fields overlay glibc's leading
 * buffer/allocated/used/syntax/fastmap/translate slots (bash never reinterprets those for a
 * compiled pattern; `regfree` frees rx_prog_ == glibc `buffer`). */
typedef struct {
  void *rx_prog_;   /* @0  (glibc buffer)    — instructions then class bitmaps */
  long rx_nprog_;   /* @8  (glibc allocated) */
  long rx_ncls_;    /* @16 (glibc used) */
  long rx_cflags_;  /* @24 (glibc syntax) */
  void *rx_pad0_;   /* @32 (glibc fastmap)   — unused */
  void *rx_pad1_;   /* @40 (glibc translate) — unused */
  long re_nsub;     /* @48 (glibc re_nsub)   — the field bash reads */
  long rx_pad2_;    /* @56 (glibc bitfields) — unused */
} regex_t;          /* 64 bytes, glibc-compatible */

typedef struct {
  int rm_so;        /* regoff_t (glibc: int) */
  int rm_eo;
} regmatch_t;

/* --- AST ------------------------------------------------------------------ */

#define RXN_CHAR 0
#define RXN_ANY 1
#define RXN_CLASS 2
#define RXN_BOL 3
#define RXN_EOL 4
#define RXN_CAT 5
#define RXN_ALT 6
#define RXN_REP 7 /* a=min, b=max (-1 = unbounded), l=body */
#define RXN_GRP 8 /* a=group index, l=body */
#define RXN_NIL 9 /* the empty regex (an empty alternation branch) */

typedef struct RxNode {
  int kind;
  long a;
  long b;
  struct RxNode *l;
  struct RxNode *r;
} RxNode;

/* --- program -------------------------------------------------------------- */

#define RXI_CHAR 0  /* a = the (folded) byte */
#define RXI_ANY 1
#define RXI_CLASS 2 /* a = bitmap index */
#define RXI_BOL 3
#define RXI_EOL 4
#define RXI_SPLIT 5 /* a, b = relative targets; a explored first (greedy) */
#define RXI_JMP 6   /* a = relative target */
#define RXI_SAVE 7  /* a = capture slot */
#define RXI_MATCH 8

typedef struct {
  int op;
  long a;
  long b;
} RxInst;

/* --- compile state -------------------------------------------------------- */

typedef struct {
  char *p;          /* parse cursor */
  RxNode *arena;    /* bump arena for nodes */
  long used;
  long cap;
  unsigned char *cls; /* bitmaps, 32 bytes each, emitted during parse */
  long ncls;
  long clscap;
  long ngrp;
  int icase;
  int err;
} Rx;

static RxNode *rx_node_(Rx *g, int kind) {
  if (g->used >= g->cap) { g->err = REG_ESPACE; return g->arena; }
  RxNode *n = g->arena + g->used;
  g->used = g->used + 1;
  n->kind = kind;
  n->a = 0;
  n->b = 0;
  n->l = 0;
  n->r = 0;
  return n;
}

static int rx_lower_(int c) {
  if (c >= 'A' && c <= 'Z') return c - 'A' + 'a';
  return c;
}
static int rx_upper_(int c) {
  if (c >= 'a' && c <= 'z') return c - 'a' + 'A';
  return c;
}

static void rx_bit_(unsigned char *m, int c, int icase) {
  m[(c >> 3) & 31] = m[(c >> 3) & 31] | (1 << (c & 7));
  if (icase) {
    int o = rx_lower_(c) == c ? rx_upper_(c) : rx_lower_(c);
    m[(o >> 3) & 31] = m[(o >> 3) & 31] | (1 << (o & 7));
  }
}

static int rx_streq_(char *a, char *b) {
  while (*a && *a == *b) { a = a + 1; b = b + 1; }
  return *a == *b;
}

/* Set the [[:name:]] class at `p` (past "[:") into `m`; leaves `*end` past ":]".
   Unknown/unterminated name → REG_BADPAT (POSIX makes it an error, unlike
   fnmatch's literal fallback). */
static int rx_class_(Rx *g, char *p, unsigned char *m, char **end) {
  char name[8];
  int n = 0;
  while (p[n] && p[n] != ':' && n < 7) { name[n] = p[n]; n = n + 1; }
  if (p[n] != ':' || p[n + 1] != ']') return REG_BADPAT;
  name[n] = 0;
  *end = p + n + 2;
  int c;
  for (c = 1; c < 256; c = c + 1) {
    int hit = 0;
    int lo = rx_lower_(c);
    if (rx_streq_(name, "alpha")) hit = lo >= 'a' && lo <= 'z';
    else if (rx_streq_(name, "digit")) hit = c >= '0' && c <= '9';
    else if (rx_streq_(name, "alnum")) hit = (lo >= 'a' && lo <= 'z') || (c >= '0' && c <= '9');
    else if (rx_streq_(name, "upper")) hit = c >= 'A' && c <= 'Z';
    else if (rx_streq_(name, "lower")) hit = c >= 'a' && c <= 'z';
    else if (rx_streq_(name, "space"))
      hit = c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
    else if (rx_streq_(name, "blank")) hit = c == ' ' || c == '\t';
    else if (rx_streq_(name, "punct"))
      hit = c > ' ' && c < 127 && !(lo >= 'a' && lo <= 'z') && !(c >= '0' && c <= '9');
    else if (rx_streq_(name, "print")) hit = c >= ' ' && c < 127;
    else if (rx_streq_(name, "graph")) hit = c > ' ' && c < 127;
    else if (rx_streq_(name, "cntrl")) hit = c < ' ' || c == 127;
    else if (rx_streq_(name, "xdigit"))
      hit = (c >= '0' && c <= '9') || (lo >= 'a' && lo <= 'f');
    else return REG_BADPAT;
    if (hit) rx_bit_(m, c, g->icase && (rx_streq_(name, "upper") || rx_streq_(name, "lower")));
  }
  return 0;
}

/* Parse a bracket expression (cursor past the `[`) into a fresh bitmap; returns
   its index or -1 on error. POSIX brackets: `^` negates first, a leading `]` is
   literal, `-` is literal first/last, `\` is NOT special. */
static long rx_bracket_(Rx *g) {
  if (g->ncls >= g->clscap) { g->err = REG_ESPACE; return -1; }
  unsigned char *m = g->cls + g->ncls * 32;
  int i;
  for (i = 0; i < 32; i = i + 1) m[i] = 0;
  int neg = 0;
  char *p = g->p;
  if (*p == '^') { neg = 1; p = p + 1; }
  int first = 1;
  while (*p && (*p != ']' || first)) {
    first = 0;
    if (p[0] == '[' && p[1] == ':') {
      char *end;
      if (rx_class_(g, p + 2, m, &end) != 0) { g->err = REG_BADPAT; return -1; }
      p = end;
      continue;
    }
    int lo = (unsigned char)*p;
    p = p + 1;
    if (*p == '-' && p[1] && p[1] != ']') {
      int hi = (unsigned char)p[1];
      p = p + 2;
      if (hi < lo) { g->err = REG_BADPAT; return -1; }
      int c;
      for (c = lo; c <= hi; c = c + 1) rx_bit_(m, c, g->icase);
    } else {
      rx_bit_(m, lo, g->icase);
    }
  }
  if (!*p) { g->err = REG_BADPAT; return -1; } /* unterminated */
  g->p = p + 1;
  if (neg) {
    for (i = 0; i < 32; i = i + 1) m[i] = ~m[i] & 255;
    m[0] = m[0] & 254; /* NUL never matches */
  }
  long idx = g->ncls;
  g->ncls = g->ncls + 1;
  return idx;
}

static RxNode *rx_alt_(Rx *g);

static RxNode *rx_atom_(Rx *g) {
  int c = (unsigned char)*g->p;
  if (c == '(') {
    g->p = g->p + 1;
    g->ngrp = g->ngrp + 1;
    long idx = g->ngrp;
    if (idx > RX_MAXSUB) { g->err = REG_ESPACE; return rx_node_(g, RXN_NIL); }
    RxNode *body = rx_alt_(g);
    if (*g->p != ')') { g->err = REG_BADPAT; return body; }
    g->p = g->p + 1;
    RxNode *n = rx_node_(g, RXN_GRP);
    n->a = idx;
    n->l = body;
    return n;
  }
  if (c == '[') {
    g->p = g->p + 1;
    long idx = rx_bracket_(g);
    RxNode *n = rx_node_(g, RXN_CLASS);
    n->a = idx < 0 ? 0 : idx;
    return n;
  }
  if (c == '.') { g->p = g->p + 1; return rx_node_(g, RXN_ANY); }
  if (c == '^') { g->p = g->p + 1; return rx_node_(g, RXN_BOL); }
  if (c == '$') { g->p = g->p + 1; return rx_node_(g, RXN_EOL); }
  if (c == '\\') {
    if (!g->p[1]) { g->err = REG_BADPAT; return rx_node_(g, RXN_NIL); }
    c = (unsigned char)g->p[1];
    g->p = g->p + 2;
    RxNode *n = rx_node_(g, RXN_CHAR);
    n->a = g->icase ? rx_lower_(c) : c;
    return n;
  }
  g->p = g->p + 1;
  RxNode *n = rx_node_(g, RXN_CHAR);
  n->a = g->icase ? rx_lower_(c) : c;
  return n;
}

/* `{n,m}` after an atom. Returns 1 and fills min/max on a valid interval;
   returns 0 (cursor untouched) when the `{` is not interval syntax — it then
   stays a literal, the common lenient reading. */
static int rx_interval_(Rx *g, long *min, long *max) {
  char *p = g->p + 1;
  if (*p < '0' || *p > '9') return 0;
  long n = 0;
  while (*p >= '0' && *p <= '9') { n = n * 10 + (*p - '0'); p = p + 1; }
  long m = n;
  if (*p == ',') {
    p = p + 1;
    if (*p == '}') {
      m = -1;
    } else {
      if (*p < '0' || *p > '9') return 0;
      m = 0;
      while (*p >= '0' && *p <= '9') { m = m * 10 + (*p - '0'); p = p + 1; }
    }
  }
  if (*p != '}') return 0;
  if (n > 255 || (m != -1 && (m > 255 || m < n))) { g->err = REG_BADPAT; return 0; }
  g->p = p + 1;
  *min = n;
  *max = m;
  return 1;
}

static RxNode *rx_rep_(Rx *g) {
  RxNode *n = rx_atom_(g);
  for (;;) {
    int c = *g->p;
    long min;
    long max;
    if (c == '*') { min = 0; max = -1; g->p = g->p + 1; }
    else if (c == '+') { min = 1; max = -1; g->p = g->p + 1; }
    else if (c == '?') { min = 0; max = 1; g->p = g->p + 1; }
    else if (c == '{' && rx_interval_(g, &min, &max)) { }
    else return n;
    RxNode *r = rx_node_(g, RXN_REP);
    r->a = min;
    r->b = max;
    r->l = n;
    n = r;
  }
}

static RxNode *rx_cat_(Rx *g) {
  int c = *g->p;
  if (!c || c == '|' || c == ')') return rx_node_(g, RXN_NIL);
  RxNode *n = rx_rep_(g);
  while (!g->err) {
    c = *g->p;
    if (!c || c == '|' || c == ')') return n;
    RxNode *r = rx_rep_(g);
    RxNode *cat = rx_node_(g, RXN_CAT);
    cat->l = n;
    cat->r = r;
    n = cat;
  }
  return n;
}

static RxNode *rx_alt_(Rx *g) {
  RxNode *n = rx_cat_(g);
  while (*g->p == '|' && !g->err) {
    g->p = g->p + 1;
    RxNode *r = rx_cat_(g);
    RxNode *alt = rx_node_(g, RXN_ALT);
    alt->l = n;
    alt->r = r;
    n = alt;
  }
  return n;
}

/* --- emit (relative jumps, so {n,m} block-copies compose) ------------------ */

static long rx_size_(RxNode *n) {
  if (n->kind == RXN_NIL) return 0;
  if (n->kind == RXN_CAT) return rx_size_(n->l) + rx_size_(n->r);
  if (n->kind == RXN_ALT) return rx_size_(n->l) + rx_size_(n->r) + 2;
  if (n->kind == RXN_GRP) return rx_size_(n->l) + 2;
  if (n->kind == RXN_REP) {
    long s = rx_size_(n->l);
    if (n->b == -1) return n->a * s + s + 2;      /* n copies + star */
    return n->a * s + (n->b - n->a) * (s + 1);    /* n copies + optionals */
  }
  return 1;
}

static long rx_emit_(RxNode *n, RxInst *o, long at) {
  if (n->kind == RXN_NIL) return at;
  if (n->kind == RXN_CAT) return rx_emit_(n->r, o, rx_emit_(n->l, o, at));
  if (n->kind == RXN_ALT) {
    long sl = rx_size_(n->l);
    long sr = rx_size_(n->r);
    o[at].op = RXI_SPLIT;
    o[at].a = 1;           /* left branch */
    o[at].b = sl + 2;      /* right branch */
    rx_emit_(n->l, o, at + 1);
    o[at + 1 + sl].op = RXI_JMP;
    o[at + 1 + sl].a = sr + 1; /* past the right branch */
    rx_emit_(n->r, o, at + 2 + sl);
    return at + 2 + sl + sr;
  }
  if (n->kind == RXN_GRP) {
    o[at].op = RXI_SAVE;
    o[at].a = 2 * n->a;
    long end = rx_emit_(n->l, o, at + 1);
    o[end].op = RXI_SAVE;
    o[end].a = 2 * n->a + 1;
    return end + 1;
  }
  if (n->kind == RXN_REP) {
    long s = rx_size_(n->l);
    long i;
    for (i = 0; i < n->a; i = i + 1) at = rx_emit_(n->l, o, at);
    if (n->b == -1) {
      /* star: SPLIT(body, out); body; JMP(back) */
      o[at].op = RXI_SPLIT;
      o[at].a = 1;
      o[at].b = s + 2;
      rx_emit_(n->l, o, at + 1);
      o[at + 1 + s].op = RXI_JMP;
      o[at + 1 + s].a = -(s + 1);
      return at + s + 2;
    }
    /* optionals: each is SPLIT(body, past-ALL-remaining-optionals) */
    long left = n->b - n->a;
    for (i = 0; i < left; i = i + 1) {
      o[at].op = RXI_SPLIT;
      o[at].a = 1;
      o[at].b = (left - i) * (s + 1);
      rx_emit_(n->l, o, at + 1);
      at = at + s + 1;
    }
    return at;
  }
  if (n->kind == RXN_CHAR) { o[at].op = RXI_CHAR; o[at].a = n->a; }
  else if (n->kind == RXN_ANY) o[at].op = RXI_ANY;
  else if (n->kind == RXN_CLASS) { o[at].op = RXI_CLASS; o[at].a = n->a; }
  else if (n->kind == RXN_BOL) o[at].op = RXI_BOL;
  else o[at].op = RXI_EOL;
  return at + 1;
}

/* --- the matcher: exhaustive, longest-at-leftmost -------------------------- */

typedef struct {
  RxInst *prog;
  unsigned char *cls;
  char *str;
  long len;
  int icase;
  int eflags;
  long steps;
  long best;
  long caps[2 * RX_MAXSUB + 2];
  long best_caps[2 * RX_MAXSUB + 2];
  long ncaps;
} RxEnv;

static void rx_run_(RxEnv *e, long pc, long sp) {
  for (;;) {
    if (e->steps <= 0) return;
    e->steps = e->steps - 1;
    int op = e->prog[pc].op;
    if (op == RXI_CHAR) {
      int c = sp < e->len ? (unsigned char)e->str[sp] : -1;
      if (e->icase) c = rx_lower_(c);
      if (c != e->prog[pc].a) return;
      pc = pc + 1;
      sp = sp + 1;
    } else if (op == RXI_ANY) {
      if (sp >= e->len) return;
      pc = pc + 1;
      sp = sp + 1;
    } else if (op == RXI_CLASS) {
      if (sp >= e->len) return;
      unsigned char c = (unsigned char)e->str[sp];
      unsigned char *m = e->cls + e->prog[pc].a * 32;
      if (!(m[c >> 3] & (1 << (c & 7)))) return;
      pc = pc + 1;
      sp = sp + 1;
    } else if (op == RXI_BOL) {
      if (sp != 0 || (e->eflags & REG_NOTBOL)) return;
      pc = pc + 1;
    } else if (op == RXI_EOL) {
      if (sp != e->len || (e->eflags & REG_NOTEOL)) return;
      pc = pc + 1;
    } else if (op == RXI_SAVE) {
      long slot = e->prog[pc].a;
      long old = e->caps[slot];
      e->caps[slot] = sp;
      rx_run_(e, pc + 1, sp);
      e->caps[slot] = old;
      return;
    } else if (op == RXI_SPLIT) {
      rx_run_(e, pc + e->prog[pc].a, sp);
      pc = pc + e->prog[pc].b;
    } else if (op == RXI_JMP) {
      pc = pc + e->prog[pc].a;
    } else { /* RXI_MATCH */
      if (sp > e->best) {
        e->best = sp;
        long i;
        for (i = 0; i < e->ncaps; i = i + 1) e->best_caps[i] = e->caps[i];
      }
      return;
    }
  }
}

/* --- the POSIX surface ----------------------------------------------------- */

int regcomp(regex_t *preg, char *pattern, int cflags) {
  if (!(cflags & REG_EXTENDED)) return REG_BADPAT; /* ERE only (bash's =~) */
  long plen = 0;
  while (pattern[plen]) plen = plen + 1;
  Rx g;
  g.p = pattern;
  g.cap = 4 * plen + 8;
  g.arena = (RxNode *)__px_malloc(0, g.cap * sizeof(RxNode));
  if (!g.arena) return REG_ESPACE;
  g.used = 0;
  g.clscap = plen + 1;
  g.cls = (unsigned char *)__px_malloc(0, g.clscap * 32);
  if (!g.cls) { __px_free(0, (long)g.arena); return REG_ESPACE; }
  g.ncls = 0;
  g.ngrp = 0;
  g.icase = (cflags & REG_ICASE) != 0;
  g.err = 0;
  RxNode *root = rx_alt_(&g);
  if (!g.err && *g.p) g.err = REG_BADPAT; /* stray `)` */
  if (g.err) {
    __px_free(0, (long)g.arena);
    __px_free(0, (long)g.cls);
    return g.err;
  }
  /* prog = SAVE 0; body; SAVE 1; MATCH — one block: instructions then bitmaps */
  long n = rx_size_(root) + 3;
  RxInst *prog = (RxInst *)__px_malloc(0, n * sizeof(RxInst) + g.ncls * 32);
  if (!prog) {
    __px_free(0, (long)g.arena);
    __px_free(0, (long)g.cls);
    return REG_ESPACE;
  }
  prog[0].op = RXI_SAVE;
  prog[0].a = 0;
  long end = rx_emit_(root, prog, 1);
  prog[end].op = RXI_SAVE;
  prog[end].a = 1;
  prog[end + 1].op = RXI_MATCH;
  unsigned char *cls = (unsigned char *)(prog + n);
  long i;
  for (i = 0; i < g.ncls * 32; i = i + 1) cls[i] = g.cls[i];
  preg->re_nsub = g.ngrp;
  preg->rx_prog_ = prog;
  preg->rx_nprog_ = n;
  preg->rx_ncls_ = g.ncls;
  preg->rx_cflags_ = cflags;
  __px_free(0, (long)g.arena);
  __px_free(0, (long)g.cls);
  return 0;
}

int regexec(regex_t *preg, char *string, long nmatch, regmatch_t *pmatch, int eflags) {
  RxEnv e;
  e.prog = (RxInst *)preg->rx_prog_;
  e.cls = (unsigned char *)(e.prog + preg->rx_nprog_);
  e.str = string;
  e.len = 0;
  while (string[e.len]) e.len = e.len + 1;
  e.icase = (preg->rx_cflags_ & REG_ICASE) != 0;
  e.eflags = eflags;
  e.ncaps = 2 * (preg->re_nsub + 1);
  long start;
  for (start = 0; start <= e.len; start = start + 1) {
    e.steps = RX_STEPS;
    e.best = -1;
    long i;
    for (i = 0; i < e.ncaps; i = i + 1) e.caps[i] = -1;
    rx_run_(&e, 0, start);
    if (e.best >= 0) {
      if (!(preg->rx_cflags_ & REG_NOSUB)) {
        for (i = 0; i < nmatch; i = i + 1) {
          if (i <= preg->re_nsub) {
            pmatch[i].rm_so = e.best_caps[2 * i];
            pmatch[i].rm_eo = e.best_caps[2 * i + 1];
          } else {
            pmatch[i].rm_so = -1;
            pmatch[i].rm_eo = -1;
          }
        }
      }
      return 0;
    }
  }
  return REG_NOMATCH;
}

void regfree(regex_t *preg) {
  if (preg->rx_prog_) __px_free(0, (long)preg->rx_prog_);
  preg->rx_prog_ = 0;
}
