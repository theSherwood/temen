
int __vm_cap_count(void);
int __vm_cap_at(int i, int *type_id_out);

/* Discover the handle of interface `want` from the domain's own capability table
   (cap.self reflection — the discovery tier IMPORTS.md keeps). */
static int __capof(int want) {
  int n = __vm_cap_count();
  int i = 0;
  while (i < n) {
    int t = 0;
    int h = __vm_cap_at(i, &t);
    if (t == want) return h;
    i = i + 1;
  }
  return -1;
}
static int __h_px = -1;
static int __px(void) { if (__h_px < 0) __h_px = __capof(13); return __h_px; }   /* HOST_FN = 13 */
static int __h_inst = -1;
static int __inst(void) { if (__h_inst < 0) __h_inst = __capof(6); return __h_inst; } /* Instantiator = 6 */

long __px_write(int cap, long fd, long buf, long len);
long __px_read(int cap, long fd, long buf, long len);
long __px_open(int cap, long path, long len, long flags);
long __px_close(int cap, long fd);
long __px_unlink(int cap, long path, long len);
long __px_getcwd(int cap, long buf, long size);
long __px_chdir(int cap, long path, long len);
long __px_getenv(int cap, long name, long len);
long __px_setenv(int cap, long name, long nlen, long val, long vlen, long overwrite);
void __px_exit(int cap, int code);
long __px_opendir(int cap, long path, long len);
long __px_readdir(int cap, long dir, long namebuf, long namecap);
long __px_closedir(int cap, long dir);
long __px_argc(int cap);
long __px_argv(int cap, long i, long buf, long cap2);
/* Personality `exec` surface (STAGE1.md §5): PATH lookup + the forwardable stdout handle. The spawn
   itself is the shell's own `Instantiator` call.cap — `__spawn` (op 13) / `__join` (op 1) — dispatched
   on the reflection-discovered `Instantiator` handle (`__inst()`), like every import here. */
long __px_exec_lookup(int cap, long name, long len);
long __px_exec_win(int cap, long module);
long __px_exec_stdout(int cap);
long __px_exec_stdin(int cap, long buf, long len);
long __spawn(int inst, long module, long gp, long gn, long entry, long off, long sl, long q);
long __join(int inst, long child);
/* Ring pipelines (STAGE1.md item 6): mint a shareable region (`AddressSpace` op 5) and alias it into
   this window (`SharedRegion` ops 0/1/3), dispatched on the reflection-discovered handles below. */
long __as_region(int cap, long len);
long __rg_map(int cap, long win_off, long region_off, long len, long prot);
long __rg_unmap(int cap, long win_off, long len);
long __rg_granule(int cap);
static int __h_as = -1;
static int __as(void) { if (__h_as < 0) __h_as = __capof(5); return __h_as; }   /* AddressSpace = 5 */

static long slen(char *s) { long n = 0; while (s[n]) n = n + 1; return n; }

long write(long fd, void *buf, long n) { return __px_write(__px(), fd, (long)buf, n); }
long read(long fd, void *buf, long n) { return __px_read(__px(), fd, (long)buf, n); }
long open(char *path, long flags) { return __px_open(__px(), (long)path, slen(path), flags); }
long close(long fd) { return __px_close(__px(), fd); }
long unlink(char *path) { return __px_unlink(__px(), (long)path, slen(path)); }
char *getcwd(char *buf, long size) { return __px_getcwd(__px(), (long)buf, size) > 0 ? buf : 0; }
long chdir(char *path) { return __px_chdir(__px(), (long)path, slen(path)); }
char *getenv(char *name) { return (char *)__px_getenv(__px(), (long)name, slen(name)); }
long setenv_(char *name, char *val) { return __px_setenv(__px(), (long)name, slen(name), (long)val, slen(val), 1); }
void exit(int code) { __px_exit(__px(), code); }
/* A DIR is just the personality's stream handle (a long); readdir writes the next name. */
long opendir(char *path) { return __px_opendir(__px(), (long)path, slen(path)); }
long readdir(long dir, char *namebuf, long cap) { return __px_readdir(__px(), dir, (long)namebuf, cap); }
long closedir(long dir) { return __px_closedir(__px(), dir); }
/* The host-side argument vector (personality extension): sh reads its own argv here. */
int argc_(void) { return (int)__px_argc(__px()); }
long getarg(int i, char *buf, long cap) { return __px_argv(__px(), i, (long)buf, cap); }
