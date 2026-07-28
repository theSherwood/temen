
#define O_WRONLY 1
#define O_CREAT  0100
#define O_TRUNC  01000
#define O_APPEND 02000
static char cwd[256];
/* Current stdio for the command in flight: `out_fd` is 1 unless a `>`/`>>` redirect is active,
   `in_fd` is 0 unless a `<` redirect is active. run_line points them at files and restores them. */
static long out_fd = 1;
static long in_fd = 0;
/* Exit status of the last command, surfaced as `$?` and consumed by `&&`/`||`. 0 = success. */
static int last_status = 0;
static int streq(char *a, char *b) { int i = 0; while (a[i] && a[i] == b[i]) i++; return a[i] == 0 && b[i] == 0; }

/* Ring pipelines (STAGE1.md item 6): when a stage-0 producer pumps into a ring, `run_line_io` gets
   `def_out = -2` (the RING_FD sentinel) and `ring_out` holds the mapped ring's window address; every
   output funnel routes through `wr_out`, so a stage's own `>` redirect still overrides the pipe
   (out_fd then points at the file, not the sentinel). A closed ring (the consumer exited early —
   `head`) swallows further output, the SIGPIPE-lite. */
static long ring_out = 0;
static int ring_out_closed = 0;
static long wr_out(char *b, long n) {
#ifndef SVM_SHELL_SEQUENTIAL
  if (out_fd == -2) {
    if (ring_out_closed) return n;
    long r = ring_write(ring_out, b, n);
    if (r <= 0) { ring_out_closed = 1; return n; }
    return r;
  }
#endif
  return write(out_fd, b, n);
}
static void puts_(char *s) { wr_out(s, slen(s)); }

/* Emit a non-negative count in decimal (wc's output). */
static void put_num(long n) {
  static char b[24]; int i = 24;
  if (n == 0) { puts_("0"); return; }
  while (n > 0) { b[--i] = '0' + (int)(n % 10); n /= 10; }
  wr_out(b + i, 24 - i);
}

/* Read source for a filter: an explicit path (caller must close), else the current in_fd. */
static long src_fd(char *path, int *close_it) {
  if (path) { *close_it = 1; return open(path, 0); }   /* O_RDONLY */
  *close_it = 0; return in_fd;
}

/* Parse a non-negative decimal prefix (head/tail counts). */
static long atoi_(char *s) {
  long v = 0; int i = 0;
  while (s[i] >= '0' && s[i] <= '9') { v = v * 10 + (s[i] - '0'); i++; }
  return v;
}

/* Substring test: does `hay` contain `needle`? An empty needle matches. */
static int contains(char *hay, char *needle) {
  for (int i = 0; hay[i]; i++) {
    int j = 0; while (needle[j] && hay[i + j] == needle[j]) j++;
    if (needle[j] == 0) return 1;
  }
  return needle[0] == 0;
}

/* Byte-wise string compare (for sort/uniq): <0, 0, >0. */
static int scmp(char *a, char *b) {
  int i = 0; while (a[i] && a[i] == b[i]) i++;
  return (int)(unsigned char)a[i] - (int)(unsigned char)b[i];
}

/* Bounded string copy (dst holds up to 255 bytes + NUL). */
static void scpy(char *d, char *s) {
  int i = 0; while (s[i] && i < 255) { d[i] = s[i]; i++; } d[i] = 0;
}

/* Read one newline-delimited line from fd into buf (NUL-terminated, newline dropped); returns its
   length, or -1 at EOF with nothing read. One byte per read keeps it correct across any source. */
static long read_line(long fd, char *buf, long lim) {
  long n = 0; char c;
  for (;;) {
    long r = read(fd, &c, 1);
    if (r <= 0) { if (n == 0) return -1; break; }
    if (c == '\n') break;
    if (n < lim - 1) buf[n++] = c;
  }
  buf[n] = 0;
  return n;
}

/* Split `line` into space-separated tokens (runs of spaces collapse), writing pointers into argv and
   returning the count (capped at MAXARGS). Mutates `line` in place with NUL terminators. The cap is
   generous so glob expansion (which grows argv) has room. */
#define MAXARGS 64
static int tokenize(char *line, char **argv) {
  int argc = 0, i = 0;
  for (;;) {
    while (line[i] == ' ') i++;
    if (line[i] == 0 || argc >= MAXARGS) break;
    argv[argc++] = line + i;
    while (line[i] && line[i] != ' ') i++;
    if (line[i] == ' ') line[i++] = 0;
  }
  return argc;
}

/* Shell variables (distinct from the personality environment until `export`ed). A tiny flat table:
   name/value pairs in fixed storage, linear-scanned. */
#define NVARS 32
static char var_name[NVARS][32];
static char var_val[NVARS][128];
static int nvars = 0;
static char *get_var(char *name) {
  for (int i = 0; i < nvars; i++)
    if (streq(var_name[i], name)) return var_val[i];
  return 0;
}
static void set_var(char *name, char *val) {
  int slot = -1;
  for (int i = 0; i < nvars; i++)
    if (streq(var_name[i], name)) { slot = i; break; }
  if (slot < 0) { if (nvars >= NVARS) return; slot = nvars++; }
  int i = 0; while (name[i] && i < 31) { var_name[slot][i] = name[i]; i++; } var_name[slot][i] = 0;
  int j = 0; while (val[j] && j < 127) { var_val[slot][j] = val[j]; j++; } var_val[slot][j] = 0;
}

/* Format a non-negative integer into one of a few rotating static buffers (for `$?` expansion). */
static char *itoa_(long n) {
  static char ring[4][24]; static int k = 0;
  char *b = ring[k]; k = (k + 1) & 3;
  char t[24]; int ti = 0;
  if (n == 0) t[ti++] = '0';
  while (n > 0) { t[ti++] = '0' + (int)(n % 10); n /= 10; }
  int bi = 0; while (ti > 0) b[bi++] = t[--ti]; b[bi] = 0;
  return b;
}

/* Expand one token: `$?` → last status, `$NAME` → shell var then environment (empty if unset), else
   the token unchanged. Returned pointers stay valid for the command's duration. */
static char *expand(char *tok) {
  if (tok[0] != '$') return tok;
  char *name = tok + 1;
  if (streq(name, "?")) return itoa_(last_status);
  char *v = get_var(name);
  if (v) return v;
  v = getenv(name);
  return v ? v : "";
}

/* Evaluate `test`/`[ … ]` and return a shell status (0 = true). Supports: a lone non-empty string;
   unary `-f`/`-d`/`-e` (file / dir / either exists), `-z`/`-n` (empty / non-empty); and binary
   `=`/`!=` (string) and `-eq`/`-ne`/`-lt`/`-gt` (numeric). Anything else is false. */
static int do_test(int argc, char **argv) {
  int top = argc;
  if (streq(argv[0], "[") && top > 1 && streq(argv[top - 1], "]")) top--;   /* drop the closing `]` */
  char **a = argv + 1;
  int n = top - 1;
  if (n == 1) return a[0][0] ? 0 : 1;
  if (n == 2) {
    if (streq(a[0], "-f")) { long fd = open(a[1], 0); if (fd >= 0) { close(fd); return 0; } return 1; }
    if (streq(a[0], "-d")) { long d = opendir(a[1]); if (d >= 0) { closedir(d); return 0; } return 1; }
    if (streq(a[0], "-e")) {
      long fd = open(a[1], 0); if (fd >= 0) { close(fd); return 0; }
      long d = opendir(a[1]); if (d >= 0) { closedir(d); return 0; }
      return 1;
    }
    if (streq(a[0], "-z")) return a[1][0] ? 1 : 0;
    if (streq(a[0], "-n")) return a[1][0] ? 0 : 1;
    return 1;
  }
  if (n == 3) {
    if (streq(a[1], "=")) return streq(a[0], a[2]) ? 0 : 1;
    if (streq(a[1], "!=")) return streq(a[0], a[2]) ? 1 : 0;
    if (streq(a[1], "-eq")) return atoi_(a[0]) == atoi_(a[2]) ? 0 : 1;
    if (streq(a[1], "-ne")) return atoi_(a[0]) != atoi_(a[2]) ? 0 : 1;
    if (streq(a[1], "-lt")) return atoi_(a[0]) < atoi_(a[2]) ? 0 : 1;
    if (streq(a[1], "-gt")) return atoi_(a[0]) > atoi_(a[2]) ? 0 : 1;
    return 1;
  }
  return 1;
}

/* Glob match with `*` (any run, incl. empty) and `?` (one char). Iterative with backtracking. */
static int fnmatch_(char *p, char *s) {
  char *star = 0, *ss = 0;
  while (*s) {
    if (*p == '?' || *p == *s) { p++; s++; }
    else if (*p == '*') { star = p++; ss = s; }
    else if (star) { p = star + 1; s = ++ss; }
    else return 0;
  }
  while (*p == '*') p++;
  return *p == 0;
}

/* Expand a glob token into matching absolute paths, appending pointers (into `store`) to `out`. The
   token is split at its last `/` into a directory (default: cwd) and a pattern; each directory entry
   matching the pattern yields `dir/name`. Returns the number of matches appended. */
static int glob_expand(char *tok, char **out, int *oc, char store[][256], int *sn, int cap) {
  int last = -1;
  for (int i = 0; tok[i]; i++) if (tok[i] == '/') last = i;
  static char dir[256]; char *pat;
  if (last < 0) { getcwd(dir, 256); pat = tok; }
  else if (last == 0) { dir[0] = '/'; dir[1] = 0; pat = tok + 1; }
  else { int k = 0; while (k < last && k < 255) { dir[k] = tok[k]; k++; } dir[k] = 0; pat = tok + last + 1; }
  long d = opendir(dir);
  if (d < 0) return 0;
  static char name[256]; int matched = 0;
  while (readdir(d, name, 256) > 0) {
    if (!fnmatch_(pat, name)) continue;
    if (*oc >= cap || *sn >= 64) break;
    char *g = store[*sn];
    int p = 0, k = 0;
    while (dir[k] && p < 254) g[p++] = dir[k++];
    if (p == 0 || g[p - 1] != '/') g[p++] = '/';
    k = 0; while (name[k] && p < 255) g[p++] = name[k++];
    g[p] = 0;
    out[(*oc)++] = g; (*sn)++; matched++;
  }
  closedir(d);
  return matched;
}

/* Execute one command after redirection has been stripped. `line` is tokenized into argv; builtins
   read their input from a path argument or, absent one, the (possibly redirected) in_fd. Returns the
   command's exit status (0 = success). */
/* Spawn an external command (STAGE1.md §5). `pool` (a big writable global) forces a window large
   enough to hold a 128 KiB-aligned 128 KiB command carve below the stack, and holds the grant record.
   The command's stdout is the personality's forwardable `Stream` (`exec_stdout`), re-granted by name so
   its `write(1, …)` reaches the shell's sink — a `>`/`|` redirect on an external command is not honored
   (that needs the Power-2 `Endpoint`, STAGE1.md); the command always writes to the terminal sink. */
#ifndef SVM_SHELL_SEQUENTIAL
/* The size a ring stage is spawned into — equal to the `__stage` runner's declared window (a §14
   child's carve must equal its declared memory). Default 18 (256 KiB), the size chibicc lands the
   runner at under the native 16 KiB data page; the browser's 64 KiB page rounds the runner up to 19
   (512 KiB), so the browser fixture builds the shell with `-D SVM_STAGE_LOG2=19` to match. */
#ifndef SVM_STAGE_LOG2
#define SVM_STAGE_LOG2 18
#endif
#define SVM_STAGE_WIN (1L << SVM_STAGE_LOG2)
/* The size an external command is spawned into — equal to the command module's declared window (a §14
   child's carve must equal its declared memory). Default 17 (128 KiB), the size chibicc lands a small
   command at under the native 16 KiB data page; the browser's 64 KiB page rounds it up to 19, so the
   browser fixture builds the shell with `-D SVM_CMD_LOG2=19` to match its 64 KiB-page commands. */
#ifndef SVM_CMD_LOG2
#define SVM_CMD_LOG2 17
#endif
#define SVM_CMD_WIN (1L << SVM_CMD_LOG2)
/* Forces a window large enough for the ring carves (3 stages × `SVM_STAGE_WIN`) + the parent's ring-0
   map slot + the spawn grant-record scratch (all kept in `pool`, below the stack), and the smaller
   128 KiB-aligned external-command carve. Only the (guarded-out) external-command / ring paths use it,
   so the sequential build omits it — its window then fits the shell's own globals + stack. */
static char pool[4 * SVM_STAGE_WIN + 131072];
static int spawn_cmd(long mod, int argc, char **argv) {
  long out = __px_exec_stdout(__px());
  long base = (long)pool;
  long carve = (base + (SVM_CMD_WIN - 1)) & ~(SVM_CMD_WIN - 1);
  /* grant record at base: {name_off, name_len, out, flags}; "stdout" name follows at base+16 */
  int *rec = (int *)base;
  rec[0] = (int)(base + 16); rec[1] = 6; rec[2] = (int)out; rec[3] = 0;
  char *nm = (char *)(base + 16);
  nm[0]='s'; nm[1]='t'; nm[2]='d'; nm[3]='o'; nm[4]='u'; nm[5]='t';
  /* the command's args buffer at carve+128 (POWERBOX_ARGS_BASE): {argc, envc=0} then packed argv */
  char *ab = (char *)(carve + 128);
  int *hdr = (int *)ab;
  hdr[0] = argc; hdr[1] = 0;
  char *p = ab + 8;
  for (int i = 0; i < argc; i++) {
    char *s = argv[i]; long L = slen(s);
    for (long k = 0; k < L; k++) *p++ = s[k];
    *p++ = 0;
  }
  long child = __spawn(__inst(), mod, base, 1, 0, carve, SVM_CMD_LOG2, 0);
  return (int)__join(__inst(), child);
}
#endif /* SVM_SHELL_SEQUENTIAL */

static int exec_line(char *line) {
  char *argv[MAXARGS];
  int argc = tokenize(line, argv);
  if (argc == 0) return 0;
  /* A lone `NAME=VALUE` (identifier before `=`) sets a shell variable, expanding the RHS. */
  if (argc == 1) {
    char *t = argv[0]; int eq = -1;
    for (int j = 0; t[j]; j++) {
      char c = t[j];
      if (c == '=') { eq = j; break; }
      if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) break;
    }
    if (eq > 0) { t[eq] = 0; set_var(t, expand(t + eq + 1)); return 0; }
  }
  /* Expand `$`-tokens in place before dispatch. */
  for (int i = 0; i < argc; i++) argv[i] = expand(argv[i]);
  /* Glob-expand any token containing `*`/`?` against the memfs; a token with no match stays literal
     (bash's default nullglob-off). Rebuild argv from the expansion. */
  {
    static char gstore[64][256];
    char *gbuf[MAXARGS];
    int gn = 0, ac2 = 0;
    for (int i = 0; i < argc && ac2 < MAXARGS; i++) {
      char *tok = argv[i];
      int has = 0;
      for (int j = 0; tok[j]; j++) if (tok[j] == '*' || tok[j] == '?') { has = 1; break; }
      if (!has) { gbuf[ac2++] = tok; continue; }
      int m = glob_expand(tok, gbuf, &ac2, gstore, &gn, MAXARGS);
      if (m == 0 && ac2 < MAXARGS) gbuf[ac2++] = tok;
    }
    for (int i = 0; i < ac2; i++) argv[i] = gbuf[i];
    argc = ac2;
  }
  if (argc == 0) return 0;
  char *cmd = argv[0];
  char *arg = argc > 1 ? argv[1] : 0;
  int st = 0;
  if (streq(cmd, "echo")) {
    for (int i = 1; i < argc; i++) {
      puts_(argv[i]);
      if (i + 1 < argc) puts_(" ");
    }
    puts_("\n");
  } else if (streq(cmd, "export")) {
    for (int i = 1; i < argc; i++) {
      char *e = argv[i]; int eq = -1;
      for (int j = 0; e[j]; j++) if (e[j] == '=') { eq = j; break; }
      if (eq >= 0) { e[eq] = 0; char *v = expand(e + eq + 1); set_var(e, v); setenv_(e, v); }
      else { char *v = get_var(e); setenv_(e, v ? v : ""); }
    }
  } else if (streq(cmd, "true")) {
    st = 0;
  } else if (streq(cmd, "false")) {
    st = 1;
  } else if (streq(cmd, "test") || streq(cmd, "[")) {
    st = do_test(argc, argv);
  } else if (streq(cmd, "pwd")) {
    if (getcwd(cwd, 256)) puts_(cwd);
    puts_("\n");
  } else if (streq(cmd, "cd")) {
    if (arg) chdir(arg);
  } else if (streq(cmd, "cat")) {
    static char buf[256];
    long r;
    if (argc > 1) {
      for (int ai = 1; ai < argc; ai++) {          /* concatenate each file argument */
        long fd = open(argv[ai], 0);
        if (fd < 0) { puts_(argv[ai]); puts_(": not found\n"); st = 1; }
        else { while ((r = read(fd, buf, 256)) > 0) wr_out(buf, r); close(fd); }
      }
    } else {
      while ((r = read(in_fd, buf, 256)) > 0) wr_out(buf, r);   /* no args: stream in_fd */
    }
  } else if (streq(cmd, "wc")) {
    int ci; long fd = src_fd(arg, &ci);
    if (fd < 0) { puts_(arg); puts_(": not found\n"); st = 1; }
    else {
      static char buf[256];
      long r, lines = 0, words = 0, bytes = 0; int inword = 0;
      while ((r = read(fd, buf, 256)) > 0) {
        for (long i = 0; i < r; i++) {
          char c = buf[i]; bytes++;
          if (c == '\n') lines++;
          if (c == ' ' || c == '\n' || c == '\t') inword = 0;
          else { if (!inword) words++; inword = 1; }
        }
      }
      if (ci) close(fd);
      put_num(lines); puts_(" "); put_num(words); puts_(" "); put_num(bytes); puts_("\n");
    }
  } else if (streq(cmd, "grep")) {
    int ai = 1, inv = 0, cnt = 0;       /* -v: invert match; -c: print only the match count */
    while (ai < argc && argv[ai][0] == '-') {
      if (streq(argv[ai], "-v")) inv = 1;
      else if (streq(argv[ai], "-c")) cnt = 1;
      ai++;
    }
    long matches = 0;
    if (ai < argc) {
      char *pat = argv[ai++];
      int ci; long fd = src_fd(ai < argc ? argv[ai] : 0, &ci);
      if (fd < 0) { puts_(argv[ai]); puts_(": not found\n"); st = 2; }
      else {
        static char lb[256];
        while (read_line(fd, lb, 256) >= 0) {
          int m = contains(lb, pat);
          if (inv) m = !m;
          if (m) { matches++; if (!cnt) { puts_(lb); puts_("\n"); } }
        }
        if (ci) close(fd);
      }
    }
    if (cnt) { put_num(matches); puts_("\n"); }
    if (st == 0 && matches == 0) st = 1;   /* grep: no match is exit 1 */
  } else if (streq(cmd, "head")) {
    int ai = 1; long n = 10;
    if (argc > ai && streq(argv[ai], "-n") && argc > ai + 1) { n = atoi_(argv[ai + 1]); ai += 2; }
    int ci; long fd = src_fd(argc > ai ? argv[ai] : 0, &ci);
    if (fd < 0) { puts_(argv[ai]); puts_(": not found\n"); st = 1; }
    else {
      static char lb[256];
      for (long k = 0; k < n && read_line(fd, lb, 256) >= 0; k++) { puts_(lb); puts_("\n"); }
      if (ci) close(fd);
    }
  } else if (streq(cmd, "tail")) {
    int ai = 1; long n = 10;
    if (argc > ai && streq(argv[ai], "-n") && argc > ai + 1) { n = atoi_(argv[ai + 1]); ai += 2; }
    if (n > 16) n = 16;   /* the ring holds at most 16 lines */
    int ci; long fd = src_fd(argc > ai ? argv[ai] : 0, &ci);
    if (fd < 0) { puts_(argv[ai]); puts_(": not found\n"); st = 1; }
    else {
      static char ring[16][256];
      long count = 0;
      while (read_line(fd, ring[count % 16], 256) >= 0) count++;
      if (ci) close(fd);
      long start = count > n ? count - n : 0;
      for (long k = start; k < count; k++) { puts_(ring[k % 16]); puts_("\n"); }
    }
  } else if (streq(cmd, "sort")) {
    int ci; long fd = src_fd(arg, &ci);
    if (fd < 0) { puts_(arg); puts_(": not found\n"); st = 1; }
    else {
      static char buf[64][256]; int n = 0;
      while (n < 64 && read_line(fd, buf[n], 256) >= 0) n++;
      if (ci) close(fd);
      for (int i = 1; i < n; i++) {          /* insertion sort (n <= 64) */
        static char key[256]; scpy(key, buf[i]);
        int j = i - 1;
        while (j >= 0 && scmp(buf[j], key) > 0) { scpy(buf[j + 1], buf[j]); j--; }
        scpy(buf[j + 1], key);
      }
      for (int i = 0; i < n; i++) { puts_(buf[i]); puts_("\n"); }
    }
  } else if (streq(cmd, "uniq")) {
    int ci; long fd = src_fd(arg, &ci);
    if (fd < 0) { puts_(arg); puts_(": not found\n"); st = 1; }
    else {
      static char cur[256], prev[256]; int have = 0;
      while (read_line(fd, cur, 256) >= 0)
        if (!have || scmp(cur, prev) != 0) { puts_(cur); puts_("\n"); scpy(prev, cur); have = 1; }
      if (ci) close(fd);
    }
  } else if (streq(cmd, "rm")) {
    for (int i = 1; i < argc; i++)
      if (unlink(argv[i]) < 0) { puts_(argv[i]); puts_(": not found\n"); st = 1; }
  } else if (streq(cmd, "ls")) {
    char *dir = arg ? arg : (getcwd(cwd, 256), cwd);
    long d = opendir(dir);
    if (d < 0) { puts_(dir); puts_(": not found\n"); st = 1; }
    else {
      static char name[128];
      while (readdir(d, name, 128) > 0) { puts_(name); puts_("\n"); }
      closedir(d);
    }
  } else if (streq(cmd, "exit")) {
    exit(argc > 1 ? (int)atoi_(argv[1]) : last_status);
  } else {
    /* Not a builtin: look the command up in the personality's PATH registry and, if found, spawn it as
       an external child (STAGE1.md §5); otherwise the classic `<cmd>: not found`. In the sequential
       build (the browser playground) there is no spawn path, so an unknown command is always
       `not found`. */
#ifndef SVM_SHELL_SEQUENTIAL
    long mod = __px_exec_lookup(__px(), (long)cmd, slen(cmd));
    if (mod < 0) { puts_(cmd); puts_(": not found\n"); st = 127; }
    else st = spawn_cmd(mod, argc, argv);
#else
    puts_(cmd); puts_(": not found\n"); st = 127;
#endif
  }
  return st;
}

/* Strip `< file`, `> file`, and `>> file` redirects out of the line, pointing in_fd/out_fd at the
   targets (or the caller-supplied `def_in`/`def_out` when a stream is not explicitly redirected) for
   the duration of the command, then restore stdio. An explicit redirect wins over the default — so a
   pipe stage that also redirects behaves like bash. `def_in`/`def_out` fds are owned by the caller
   and are not closed here. Multiple redirects on one line are honored (`wc < in > out`). */
static int run_line_io(char *line, long def_in, long def_out) {
  long ifd = def_in, ofd = def_out, ci = -1, co = -1;
  int cmd_end = -1, i = 0, st = 1;
  while (line[i]) {
    char op = line[i];
    if (op != '<' && op != '>') { i++; continue; }
    if (cmd_end < 0) cmd_end = i;
    line[i++] = 0;                       /* terminate the token to the left of the operator */
    int append = 0;
    if (op == '>' && line[i] == '>') { append = 1; i++; }
    while (line[i] == ' ') i++;          /* skip spaces before the filename */
    char *target = line + i;
    while (line[i] && line[i] != ' ' && line[i] != '<' && line[i] != '>') i++;
    char save = line[i]; line[i] = 0;    /* terminate the filename for open() */
    if (op == '<') {
      long fd = open(target, 0);         /* O_RDONLY */
      if (fd < 0) { puts_(target); puts_(": not found\n"); goto done; }
      ifd = fd; ci = fd;
    } else {
      long flags = append ? (O_WRONLY | O_CREAT | O_APPEND) : (O_WRONLY | O_CREAT | O_TRUNC);
      long fd = open(target, flags);
      if (fd < 0) { puts_(target); puts_(": cannot open\n"); goto done; }
      ofd = fd; co = fd;
    }
    line[i] = save;                      /* restore so scanning continues past the filename */
    if (save == 0) break;
  }
  if (cmd_end >= 0) { int e = cmd_end; while (e > 0 && line[e - 1] == ' ') line[--e] = 0; }
  in_fd = ifd; out_fd = ofd;
  st = exec_line(line);
done:
  if (ci >= 0) close(ci);
  if (co >= 0) close(co);
  in_fd = 0; out_fd = 1;
  return st;
}

/* A command with its own redirects but default stdin/stdout. */
static int run_line(char *line) { return run_line_io(line, 0, 1); }

#ifndef SVM_SHELL_SEQUENTIAL
/* ---- Ring pipelines (STAGE1.md item 6): concurrent children over SharedRegion rings ---- */

/* Is stage text `st` (a stage AFTER the first) a pure ring filter — runnable by the `__stage`
   runner, which has stdin (its ring), stdout, argv, and nothing else (no memfs, no vars)? First
   token in the known filter set; no redirects/vars/globs anywhere; no file arguments (`cat`/`wc`/
   `sort`/`uniq` bare; `grep` flags + exactly a pattern; `head`/`tail` bare or `-n N`). */
static int ring_filter_ok(char *st) {
  for (int i = 0; st[i]; i++) {
    char c = st[i];
    if (c == '<' || c == '>' || c == '$' || c == '*' || c == '?') return 0;
  }
  static char cp[256]; scpy(cp, st);
  char *av[MAXARGS]; int ac = tokenize(cp, av);
  if (ac == 0) return 0;
  char *c0 = av[0];
  if (streq(c0, "cat") || streq(c0, "wc") || streq(c0, "sort") || streq(c0, "uniq")) return ac == 1;
  if (streq(c0, "grep")) {
    int ai = 1;
    while (ai < ac && av[ai][0] == '-') ai++;
    return ai < ac && ai + 1 == ac;      /* a pattern and nothing after it */
  }
  if (streq(c0, "head") || streq(c0, "tail")) {
    if (ac == 1) return 1;
    return ac == 3 && streq(av[1], "-n");
  }
  return 0;
}

/* The pool layout for a ring pipeline: 256 KiB-aligned stage carves (one per child — concurrent,
   so they must be distinct; the `__stage` runner declares memory 18), then the parent's ring-0 map
   slot (256 KiB-aligned ⇒ granule-aligned), then the spawn grant-record scratch — kept OUTSIDE
   every carve, because records are (re)written while earlier children are already running in
   theirs. */
static long stage_carve0(void) { return ((long)pool + (SVM_STAGE_WIN - 1)) & ~(SVM_STAGE_WIN - 1); }
static long stage_mapslot(void) { return stage_carve0() + 3 * SVM_STAGE_WIN; }
static long stage_records(void) { return stage_carve0() + 3 * SVM_STAGE_WIN + 65536; }

/* Spawn one ring stage: the `__stage` runner in its own 128 KiB carve, granted the shell's stdout,
   its input ring `rin`, and (for a non-final stage) its output ring `rout`; argv = the stage's
   tokens (argv[0] picks the filter). Returns the op-13 child handle. */
static long spawn_stage(long mod, char *stage, long carve, int rin, int rout) {
  static char cp[256]; scpy(cp, stage);
  char *av[MAXARGS]; int ac = tokenize(cp, av);
  long base = stage_records();
  int *rec = (int *)base;
  char *nm = (char *)(base + 64);
  long out = __px_exec_stdout(__px());
  rec[0] = (int)(base + 64); rec[1] = 6; rec[2] = (int)out; rec[3] = 0;
  nm[0]='s'; nm[1]='t'; nm[2]='d'; nm[3]='o'; nm[4]='u'; nm[5]='t';
  rec[4] = (int)(base + 70); rec[5] = 3; rec[6] = rin; rec[7] = 0;
  nm[6]='r'; nm[7]='i'; nm[8]='n';
  long gn = 2;
  if (rout >= 0) {
    rec[8] = (int)(base + 73); rec[9] = 4; rec[10] = rout; rec[11] = 0;
    nm[9]='r'; nm[10]='o'; nm[11]='u'; nm[12]='t';
    gn = 3;
  }
  /* the runner's args buffer at carve+128 (POWERBOX_ARGS_BASE): {argc, envc=0} + packed argv */
  char *ab = (char *)(carve + 128);
  int *hdr = (int *)ab;
  hdr[0] = ac; hdr[1] = 0;
  char *p = ab + 8;
  for (int i = 0; i < ac; i++) {
    char *sv = av[i]; long L = slen(sv);
    for (long k = 0; k < L; k++) *p++ = sv[k];
    *p++ = 0;
  }
  return __spawn(__inst(), mod, base, gn, 0, carve, SVM_STAGE_LOG2, 0);
}

static int run_line_io(char *line, long def_in, long def_out);

/* The concurrent pipeline: mint ns-1 regions, spawn stages 1..ns-1 as `__stage` children wired
   ring→ring→stdout, run stage 0 IN the shell (full builtin power — files, redirects) pumping into
   ring 0 through the mapped alias, mark it done, and join every child; the status is the last
   stage's, as in bash. The children run concurrently on both backends (interp fibers / JIT OS
   threads), parking on the ring futexes — real streaming with backpressure, not temp files. */
static int run_ring_pipeline(char **stages, int ns, long mod) {
  long c0 = stage_carve0();
  long mapoff = stage_mapslot();
  int nr = ns - 1;
  int rh[3];
  for (int r = 0; r < nr; r++) rh[r] = (int)__as_region(__as(), 65536);
  long g = __rg_granule(rh[0]);
  if (g <= 64 || __rg_map(rh[0], mapoff, 0, g, 3) != 0) return 125;   /* loudly, never silently */
  rcap = g - 64;
  long child[3];
  for (int s = 1; s < ns; s++) {
    int rin = rh[s - 1];
    int rout = s < ns - 1 ? rh[s] : -1;
    child[s - 1] = spawn_stage(mod, stages[s], c0 + (long)(s - 1) * SVM_STAGE_WIN, rin, rout);
  }
  ring_out = mapoff;
  ring_out_closed = 0;
  run_line_io(stages[0], 0, -2);      /* RING_FD: unredirected stage-0 output pumps the ring */
  ring_done(ring_out);
  ring_out = 0;
  int st = 0;
  for (int s = 1; s < ns; s++) st = (int)__join(__inst(), child[s - 1]);
  __rg_unmap(rh[0], mapoff, g);
  return st;
}
#endif /* SVM_SHELL_SEQUENTIAL */

/* Run a pipeline `A | B | C`. When every stage after the first is a pure filter and the `__stage`
   runner is on PATH, the stages run **concurrently** — stage 0 in the shell, the rest as spawned
   children piped through `SharedRegion` rings (STAGE1.md item 6). Otherwise (redirects/vars/globs/
   file args in a later stage, no runner registered, or >4 stages) fall back to the sequential
   memfs-temp staging: each stage's stdout lands in a temp file the next stage reads as stdin —
   same pipeline *semantics*, no concurrency. The exit status is the last stage's either way. */
static int run_pipeline(char *seg) {
  char *stages[8];
  int ns = 0, i = 0, start = 0;
  for (;;) {
    char c = seg[i];
    if (c == '|' && ns < 7) { seg[i] = 0; stages[ns++] = seg + start; start = i + 1; i++; }
    else if (c == 0) { stages[ns++] = seg + start; break; }
    else i++;
  }
  if (ns == 1) return run_line(stages[0]);
#ifndef SVM_SHELL_SEQUENTIAL
  if (ns <= 4) {
    int ok = 1;
    for (int s = 1; s < ns; s++) if (!ring_filter_ok(stages[s])) { ok = 0; break; }
    if (ok) {
      static char sn[8];
      sn[0]='_'; sn[1]='_'; sn[2]='s'; sn[3]='t'; sn[4]='a'; sn[5]='g'; sn[6]='e'; sn[7]=0;
      long mod = __px_exec_lookup(__px(), (long)sn, 7);
      if (mod >= 0) return run_ring_pipeline(stages, ns, mod);
    }
  }
#endif
  static char tmp[8];                    /* "/.pipeN" — one name per producing stage */
  tmp[0] = '/'; tmp[1] = '.'; tmp[2] = 'p'; tmp[3] = 'i'; tmp[4] = 'p'; tmp[5] = 'e'; tmp[7] = 0;
  long prev_in = 0;                      /* stage 0 reads real stdin */
  int st = 0;
  for (int s = 0; s < ns; s++) {
    long def_out = 1, tmpfd = -1;
    if (s + 1 < ns) {
      tmp[6] = '0' + s;
      tmpfd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC);
      def_out = tmpfd;
    }
    st = run_line_io(stages[s], prev_in, def_out);
    if (tmpfd >= 0) close(tmpfd);
    if (prev_in > 0) close(prev_in);     /* done reading the previous stage's temp */
    if (s + 1 < ns) { tmp[6] = '0' + s; prev_in = open(tmp, 0); }   /* next stage reads it */
  }
  if (prev_in > 0) close(prev_in);
  for (int s = 0; s + 1 < ns; s++) { tmp[6] = '0' + s; unlink(tmp); }
  return st;
}

/* Run a list of commands joined by `;`, `&&`, `||`, short-circuiting on `last_status`: `&&` runs the
   next segment only after success (0), `||` only after failure, `;` always. A skipped segment leaves
   `last_status` unchanged so it propagates down a chain, matching bash. Returns the final status. */
static int run_list(char *line) {
  int i = 0, start = 0, skip = 0;
  for (;;) {
    char c = line[i];
    int is_semi = c == ';';
    int is_and = c == '&' && line[i + 1] == '&';
    int is_or = c == '|' && line[i + 1] == '|';
    if (c == 0 || is_semi || is_and || is_or) {
      char save = c; line[i] = 0;
      if (!skip) last_status = run_pipeline(line + start);
      if (save == 0) break;
      if (is_and) skip = last_status != 0;
      else if (is_or) skip = last_status == 0;
      else skip = 0;                     /* `;` starts a fresh short-circuit context */
      i += is_semi ? 1 : 2;
      start = i;
    } else {
      i++;
    }
  }
  return last_status;
}

static char *ltrim(char *s) { while (*s == ' ') s++; return s; }
/* Does `w` start with the whole word `kw` (followed by a space or end)? */
static int kw_is(char *w, char *kw) {
  int i = 0; while (kw[i]) { if (w[i] != kw[i]) return 0; i++; }
  return w[i] == 0 || w[i] == ' ';
}

/* Single-line `if COND; then BODY…; [else BODY…;] fi`. Split on `;` into segments: the first holds
   `if COND`, later ones begin with `then`/`else` (whose remainder is the first body command) or are
   further body commands, ending at `fi`. Evaluate COND with run_list, then run the taken branch's
   commands. Returns the branch's status (0 when no branch runs). */
static int run_if(char *line) {
  char *seg[32]; int ns = 0, i = 0, start = 0;
  for (;;) {
    char c = line[i];
    if (c == ';' || c == 0) {
      line[i] = 0;
      if (ns < 32) seg[ns++] = line + start;
      if (c == 0) break;
      start = i + 1;
    }
    i++;
  }
  char *cond = ltrim(ltrim(seg[0]) + 2);   /* drop leading "if" */
  char *thenb[16]; int nt = 0;
  char *elseb[16]; int ne = 0;
  int mode = 0;                            /* 1 = collecting then-body, 2 = else-body */
  for (int s = 1; s < ns; s++) {
    char *w = ltrim(seg[s]);
    if (kw_is(w, "fi")) break;
    if (kw_is(w, "then")) { mode = 1; char *b = ltrim(w + 4); if (b[0] && nt < 16) thenb[nt++] = b; }
    else if (kw_is(w, "else")) { mode = 2; char *b = ltrim(w + 4); if (b[0] && ne < 16) elseb[ne++] = b; }
    else if (mode == 1 && nt < 16) thenb[nt++] = w;
    else if (mode == 2 && ne < 16) elseb[ne++] = w;
  }
  int rc = 0;
  if (run_list(cond) == 0) { for (int k = 0; k < nt; k++) rc = run_list(thenb[k]); }
  else { for (int k = 0; k < ne; k++) rc = run_list(elseb[k]); }
  return rc;
}

/* Top-level line dispatch: an `if …` line runs the conditional construct, everything else is a
   command list. */
/* Strip an unquoted `#` comment (bash's word-start rule): a `#` at line start or after whitespace
   begins a comment to end-of-line; a `#` mid-word (`a#b`) is kept. Truncates `line` in place. */
static void strip_comment(char *line) {
  for (int i = 0; line[i]; i++) {
    if (line[i] == '#' && (i == 0 || line[i - 1] == ' ' || line[i - 1] == '\t')) {
      line[i] = 0;
      return;
    }
  }
}

static int run_top(char *line) {
  strip_comment(line);
  char *t = ltrim(line);
  if (*t == 0) return last_status; /* a blank or comment-only line is a no-op; `$?` is unchanged */
  if (kw_is(t, "if")) return run_if(t);
  return run_list(line);
}

int main(void) {
  static char cmd[256];
  /* `sh -c "<command>"` — a single command line delivered via argv. */
  if (argc_() >= 3) {
    static char flag[8];
    if (getarg(1, flag, 8) > 0 && streq(flag, "-c") && getarg(2, cmd, 256) > 0) {
      return run_top(cmd);
    }
  }
  /* Otherwise: a read-eval loop over stdin. */
  static char line[256];
  for (;;) {
    int n = 0;
    for (;;) {
      char c;
      long r = read(0, &c, 1);
      if (r <= 0) { if (n == 0) return 0; break; }   /* EOF ends the shell */
      if (c == '\n') break;
      if (n < 255) line[n++] = c;
    }
    line[n] = 0;
    last_status = run_top(line);
  }
}
