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
long __vm_exec_module(long mod, long grants, long n, long entry, long sl);

static long xs_len_(char *s) {
  long n = 0;
  while (s[n]) n = n + 1;
  return n;
}

int execve(char *path, char **argv, char **envp) {
  long m = __px_exec_resolve(0, (long)path, xs_len_(path));
  if (m < 0) return (int)m;
  /* Pack the args region: argc/envc header at 128, strings NUL-packed at 136.
     Bounds-checked against the region end (16384) — overflow is -E2BIG with
     nothing exec'd (the region is scratch until __vm_exec_module commits). */
  int *hdr = (int *)128;
  char *s = (char *)136;
  char *end = (char *)16384;
  long argc = 0;
  long envc = 0;
  long i;
  for (i = 0; argv && argv[i]; i = i + 1) argc = argc + 1;
  for (i = 0; envp && envp[i]; i = i + 1) envc = envc + 1;
  for (i = 0; i < argc + envc; i = i + 1) {
    char *p = i < argc ? argv[i] : envp[i - argc];
    while (*p) {
      if (s >= end - 1) return -7; /* E2BIG */
      *s = *p;
      s = s + 1;
      p = p + 1;
    }
    *s = 0;
    s = s + 1;
  }
  hdr[0] = (int)argc;
  hdr[1] = (int)envc;
  /* Empty grant list (v1 self-contained commands); the window hint is advisory
     since #773 — the command runs in the caller's window if it fits. */
  __vm_exec_module(m, 0, 0, 0, 17);
  return -22; /* only on failure: the core refused (-EINVAL) and we still run */
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
