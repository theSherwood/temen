/* fork_shim.c — a minimal POSIX process libc over the svm fork/exec/wait surface (FORK.md §8.6).
 *
 * This is the guest-side layer a shell (or any fork/exec program) links so it writes the idiomatic
 *   pid = fork(); if (pid == 0) execvp(cmd, argv); else wait_pid(pid);
 * instead of hand-marshalling the §3e powerbox args buffer and driving the raw self-ops. It rides:
 *   - the `__fork`/`__wait` **offers** (granted into the powerbox by name — the manager topology), and
 *   - the `__vm_resolve` / `__vm_exec_module` / `__vm_setpgid` **self-ops** (chibicc builtins).
 *
 * Every entry point is `static`, so chibicc's dead-code pass drops the ones a program never calls: a
 * *command* that only reads its environment pulls in `getenv`/`strlen`, NOT `fork`/`execvp` — so its
 * module never carries the `__fork`/`__wait` offer imports it was not granted. #include this directly.
 */

/* Self-namespace builtins (recognized by the frontend; no grant needed — they are `cap.self` ops). */
long __vm_resolve(const char *name, long len);
long __vm_exec_module(long mod, long grants, long n, long entry, long sl);
long __vm_setpgid(long pid, long pgid);
/* The fork/wait offers — bound by the manifest to the granted `__fork`/`__wait` powerbox entries. */
long __fork(int h, long a);
long __wait(int h, long pid);

/* The current environment (NUL-terminated `char*` vector). crt convention: a `main(argc, argv, envp)`
 * command sets `environ = envp`; a shell points it at its own env before `execvp` packs it. */
char **environ;

static inline long strlen(const char *s) {
  long n = 0;
  while (s[n]) n = n + 1;
  return n;
}

/* getenv(name): the value after '=' of the first `environ` entry whose key is `name`, else 0. */
static inline char *getenv(const char *name) {
  if (!environ) return 0;
  long nl = strlen(name), i = 0;
  while (environ[i]) {
    char *e = environ[i];
    long k = 0;
    while (k < nl && e[k] == name[k]) k = k + 1;
    if (k == nl && e[k] == '=') return e + k + 1;
    i = i + 1;
  }
  return 0;
}

/* fork(): the twin's pid in the parent, 0 in the child, <0 on the serve/park race (retry). */
static inline long fork(void) { return __fork(0, 0); }

/* wait_pid(pid): block for a child's exit status, retrying the `-EAGAIN` (-11) serve/park race. `pid`
 * takes the POSIX `waitpid` selector: `-1` any child, `< -1` the process group `|pid|`, `> 0` a pid. */
static inline long wait_pid(long pid) {
  long st = __wait(0, pid);
  while (st == -11) st = __wait(0, pid);
  return st;
}

/* setpgid(pid, pgid): put child `pid` in process group `pgid` (`0` → its own leader). 0 / -errno. */
static inline int setpgid(long pid, long pgid) { return (int)__vm_setpgid(pid, pgid); }

struct __grant { int name_off; int name_len; int handle; int pad; };
static char __so_name[] = "stdout";
static struct __grant __grec;

/* execvp(file, argv): seed the §3e args buffer from a NUL-terminated `argv[]` and the current
 * `environ`, resolve the command module named `file`, inherit `stdout`, and image-replace into it.
 * Returns -1 only on failure (POSIX `execvp` returns only when the exec fails). */
static inline int execvp(const char *file, char **argv) {
  int *hdr = (int *)128;   /* POWERBOX_ARGS_BASE: {argc:u32, envc:u32} */
  char *s = (char *)136;   /* packed NUL-terminated strings: argv[0..argc) then env[0..envc) */
  long i = 0, argc = 0, envc = 0, k;
  while (argv[argc]) {
    char *a = argv[argc];
    for (k = 0; a[k]; k = k + 1) { s[i] = a[k]; i = i + 1; }
    s[i] = 0; i = i + 1;
    argc = argc + 1;
  }
  if (environ) {
    while (environ[envc]) {
      char *e = environ[envc];
      for (k = 0; e[k]; k = k + 1) { s[i] = e[k]; i = i + 1; }
      s[i] = 0; i = i + 1;
      envc = envc + 1;
    }
  }
  hdr[0] = (int)argc;
  hdr[1] = (int)envc;
  long mod = __vm_resolve(file, strlen(file));
  if (mod < 0) return -1;
  long soh = __vm_resolve(__so_name, 6);
  __grec.name_off = (int)(long)__so_name;
  __grec.name_len = 6;
  __grec.handle = (int)soh;
  __vm_exec_module(mod, (long)&__grec, 1, 0, 17);
  return -1;
}
