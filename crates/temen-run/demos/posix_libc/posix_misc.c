/* putenv / wait3 / wait4 as real guest libc — #800 (bash umbrella #794).
 *
 * Thin compositions over personality ops that already exist: putenv splits
 * `KEY=VALUE` and forwards to setenv (op 12) — a bare `KEY` removes it via
 * unsetenv (op 35), the glibc behavior bash relies on. One deliberate
 * divergence: POSIX putenv aliases the caller's string into environ, while the
 * personality's env map copies — an env var later mutated through the caller's
 * buffer won't change, which no shell pattern depends on (bash re-putenvs).
 * wait3/wait4 forward to waitpid (op 28) — including its #799 blocking path —
 * and zero the caller's rusage, since the personality meters fuel, not
 * rusage-shaped time (all-zero is the POSIX-legal "no information" answer).
 *
 * Self-contained (freestanding, own `__px_` externs — duplicate identical
 * declarations across concatenated modules are fine) for the c_posix harness.
 */

long __px_setenv(int cap, long name, long nlen, long val, long vlen, long overwrite);
long __px_unsetenv(int cap, long name, long nlen);
long __px_waitpid(int cap, long pid, long status, long opts);

static long pxm_slen_(char *s) {
  long n = 0;
  while (s[n]) n = n + 1;
  return n;
}

int putenv(char *string) {
  long i = 0;
  while (string[i] && string[i] != '=') i = i + 1;
  if (!string[i])
    return (int)__px_unsetenv(0, (long)string, i);
  return (int)__px_setenv(0, (long)string, i, (long)(string + i + 1),
                          pxm_slen_(string + i + 1), 1);
}

/* The personality has no rusage accounting (fuel is the meter); zero the whole
   struct — 144 bytes covers glibc's struct rusage (18 longs). */
static void pxm_zero_rusage_(void *ru) {
  char *p = (char *)ru;
  int i;
  for (i = 0; i < 144; i = i + 1) p[i] = 0;
}

long wait4(long pid, int *status, int options, void *rusage) {
  if (rusage) pxm_zero_rusage_(rusage);
  return __px_waitpid(0, pid, (long)status, options);
}

long wait3(int *status, int options, void *rusage) {
  return wait4(-1, status, options, rusage);
}
