/* execve/execv/execvp as real guest libc — #801 slice A (bash umbrella #794).
 *
 * The same zero-new-core-surface shape as the pipe unification: the personality
 * supplies only the path→module *policy* (`exec_resolve`, op 53 — handles were
 * pre-granted at registration, so the dispatch is bookkeeping), and the guest
 * itself performs the mechanism — the `CAP_SELF_EXEC` image-replace
 * (`__vm_exec_module`), which swaps this vCPU's image for the command's in
 * place, keeping the TaskId, so a parent's `waitpid` reaps the command's exit
 * exactly as POSIX `execve` keeps the pid. argv/envp ride the powerbox args
 * region (`[128, 16384)`: `{argc:i32, envc:i32}` then NUL-packed strings),
 * which the image-replace preserves and the new image's `_start` parses.
 *
 * POSIX shape: `execve` returns only on failure — `-ENOENT` (nothing at the
 * path), `-EACCES` (a file without the executable registration), `-E2BIG`
 * (args overflow the region), or the core's refusal (`-EINVAL`: bad module,
 * unclean context). `execvp` walks `PATH` (default `/bin`), remembering an
 * `-EACCES` while continuing on `-ENOENT`, per POSIX. v1 commands are
 * self-contained modules (empty grant list); carrying grants — the
 * personality binding itself, adopted pipe ends — is the #972-slice-2 carry,
 * gated on the import-resolver sketch on #801.
 *
 * Self-contained (freestanding, own externs) for the c_posix harness.
 */

long __px_exec_resolve(int cap, long path, long len);
long __px_getenv(int cap, long name, long len);
long __px_open(int cap, long path, long len, long flags);
long __px_read(int cap, long fd, long buf, long len);
long __px_close(int cap, long fd);
long __vm_exec_module(long mod, long grants, long n, long entry, long sl);

static long xs_len_(char *s) {
  long n = 0;
  while (s[n]) n = n + 1;
  return n;
}

int execve(char *path, char **argv, char **envp);

/* `#!` scripts (#801, one level per POSIX): a memfs file whose first bytes are `#!` re-execs its
   interpreter with the script path spliced into argv (`interp [arg] path argv[1..]`). v1
   divergence, documented: ANY memfs file with a `#!` line runs this way (the memfs has no chmod
   yet — the exec bit gates only module executables). The depth guard refuses a script
   interpreter (POSIX: the interpreter must be a real executable). */
static int xs_script_depth_;
static int xs_try_script_(char *path, char **argv, char **envp) {
  if (xs_script_depth_) return -13;
  long fd = __px_open(0, (long)path, xs_len_(path), 0);
  if (fd < 0) return -13;
  char line[128];
  long n = __px_read(0, fd, (long)line, 127);
  __px_close(0, fd);
  if (n < 3 || line[0] != '#' || line[1] != '!') return -13;
  line[n] = 0;
  long i = 2;
  while (line[i] == ' ') i = i + 1;
  char *interp = line + i;
  while (line[i] && line[i] != '\n' && line[i] != ' ') i = i + 1;
  char *arg = 0;
  if (line[i] == ' ') {
    line[i] = 0;
    i = i + 1;
    while (line[i] == ' ') i = i + 1;
    if (line[i] && line[i] != '\n') {
      arg = line + i;
      while (line[i] && line[i] != '\n') i = i + 1;
    }
  }
  line[i] = 0;
  if (!*interp) return -13;
  char *nav[64];
  long k = 0;
  nav[k] = interp; k = k + 1;
  if (arg && *arg) { nav[k] = arg; k = k + 1; }
  nav[k] = path; k = k + 1;
  long j = 1;
  while (argv && argv[j] && k < 62) { nav[k] = argv[j]; k = k + 1; j = j + 1; }
  nav[k] = 0;
  xs_script_depth_ = 1;
  int r = execve(interp, nav, envp);
  xs_script_depth_ = 0;
  return r;
}

/* #1059/#1094 NULL guard: chibicc lays the powerbox args region one 16 KiB guard up
   (`temen_ir::module_args_base` == guard + 128, the unconditional guarded layout), so a
   command reads argv from here and the host preserves this shifted range across the exec
   image-replace (`commit_fresh_image`). */
#define PX_ARGS_BASE (16384 + 128) /* POWERBOX_NULL_GUARD + POWERBOX_ARGS_BASE */

/* The args region `[PX_ARGS_BASE, PX_ARGS_BASE + 16256)` is not private to exec: a NON-child
   module's own data legitimately starts right after the reserved low region (only child-entry
   images reserve the whole args span), and the strings argv points at may themselves live INSIDE
   it (a command re-execing with argv into its own region). Packing in place therefore tramples
   the very bytes still being read — the original in-place pack survived every
   short-argv witness by byte-count luck and corrupted longer ones mid-loop. So:
   STAGE the pack in private scratch (every source string is read before any region
   byte is written), SAVE the caller's bytes under the region, splash at the point
   of no return, and RESTORE on a refused exec — POSIX: execve returns only on
   failure, and a failed exec must leave the caller intact. */
static char xs_pk_[16248]; /* staged {strings} — 16384 - 136 */
static char xs_sv_[16256]; /* caller bytes under [128, 128+8+packed), restored on refusal */

int execve(char *path, char **argv, char **envp) {
  long m = __px_exec_resolve(0, (long)path, xs_len_(path));
  if (m == -13) return xs_try_script_(path, argv, envp);
  if (m < 0) return (int)m;
  long argc = 0;
  long envc = 0;
  long i;
  for (i = 0; argv && argv[i]; i = i + 1) argc = argc + 1;
  for (i = 0; envp && envp[i]; i = i + 1) envc = envc + 1;
  /* Stage: NUL-packed strings into scratch; -E2BIG on overflow, nothing written. */
  long s = 0;
  for (i = 0; i < argc + envc; i = i + 1) {
    char *p = i < argc ? argv[i] : envp[i - argc];
    while (*p) {
      if (s >= 16247) return -7; /* E2BIG */
      xs_pk_[s] = *p;
      s = s + 1;
      p = p + 1;
    }
    xs_pk_[s] = 0;
    s = s + 1;
  }
  /* Save the caller's region bytes, splash header + strings, exec. */
  char *reg = (char *)PX_ARGS_BASE;
  for (i = 0; i < s + 8; i = i + 1) xs_sv_[i] = reg[i];
  int *hdr = (int *)PX_ARGS_BASE;
  hdr[0] = (int)argc;
  hdr[1] = (int)envc;
  for (i = 0; i < s; i = i + 1) reg[8 + i] = xs_pk_[i];
  /* Empty grant list (v1 self-contained commands); the window hint is advisory
     since #773 — the command runs in the caller's window if it fits. */
  __vm_exec_module(m, 0, 0, 0, 17);
  /* Only reached on refusal (-EINVAL): the caller keeps running — restore the
     bytes the splash covered so its own data (if any lived there) is intact. */
  for (i = 0; i < s + 8; i = i + 1) reg[i] = xs_sv_[i];
  return -22;
}

int execv(char *path, char **argv) { return execve(path, argv, 0); }

int execvp(char *file, char **argv) {
  long i = 0;
  while (file[i] && file[i] != '/') i = i + 1;
  if (file[i]) return execve(file, argv, 0); /* a path with '/' skips PATH */
  char *path = (char *)__px_getenv(0, (long)"PATH", 4);
  if (!path) path = "/bin";
  char buf[256];
  int saw_eacces = 0;
  while (*path) {
    long d = 0;
    while (path[d] && path[d] != ':') d = d + 1;
    long fl = xs_len_(file);
    if (d + 1 + fl < 255) {
      long j;
      for (j = 0; j < d; j = j + 1) buf[j] = path[j];
      buf[d] = '/';
      for (j = 0; j <= fl; j = j + 1) buf[d + 1 + j] = file[j];
      int r = execve(buf, argv, 0);
      if (r == -13) saw_eacces = 1; /* EACCES: remember, keep searching */
      else if (r != -2) return r;   /* not ENOENT: a real failure (or E2BIG) */
    }
    path = path + d;
    if (*path == ':') path = path + 1;
  }
  return saw_eacces ? -13 : -2;
}
