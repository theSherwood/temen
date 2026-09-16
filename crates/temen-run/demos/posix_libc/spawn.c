/* spawn.c — the §14 spawn as a C call (#1509): fill the `Instantiator.instantiate_rec` (op 17)
 * config record and its named-grant list, spawn, join.
 *
 * Per-child attenuation is the grant list: a parent narrows handles in its own table first
 * (`AddressSpace.sub`, `Budget.split`, an exported wrapper) and lists a different subset for each
 * child — the child's powerbox is exactly what it is handed here (DESIGN.md §3c "Attenuation needs
 * no new IR"). This file only removes the byte-laying: the 56-byte record (`temen_ir::SpawnRec`) and
 * the `{name_off, name_len, handle, flags}` grant records that every C consumer used to write by hand.
 *
 * Reaches the Instantiator through two frontend builtins (`__vm_instantiate_rec` = op 17,
 * `__vm_instantiate_join` = op 1 — static `call.cap`s like `__vm_exec_module`, since an executor op
 * reaches its seam only through a static `call.cap`) on the handle the guest discovers itself via
 * `cap.self` reflection — no new IR op.
 *
 * Record contract (little-endian, window-relative pointers; fails closed on any other version):
 *   { version: u32 = 0, entry: u32, off: u64, size_log2: u32, pager: u32 = MAX (none),
 *     module: i32 (-1 = self), budget: i32 = 0 (quota-funded), quota: i64,
 *     grants_ptr: u64, grants_n: u64 }
 * followed here by `n` × 16-byte grant records. `scratch` must be 8-byte aligned and hold
 * `56 + 16 * n` bytes; the names are referenced in place (window pointers), not copied.
 *
 * Self-contained (freestanding, own externs) for the chibicc harnesses; every helper is
 * `static inline` so the frontend's dead-code pass drops what a program never calls.
 */

long __vm_instantiate_rec(int inst, long rec);    /* Instantiator op 17 -> child handle | -errno */
long __vm_instantiate_join(int inst, long child); /* Instantiator op 1 -> the child's entry result */
int __vm_cap_count(void);
int __vm_cap_at(int i, int *type_id_out);

/* One named grant: the child resolves `name` (`self.resolve` / a named import) to a re-grant of the
 * parent's `handle`. The handle must be one this domain holds and that the host can re-grant
 * (a stream, pipe end, region, offer, forkable host proc, Module, Jit); a forged or non-grantable
 * one fails the whole spawn closed. */
typedef struct {
  char *name;
  int handle;
} vm_grant;

static inline long vm_strlen_(char *s) {
  long n = 0;
  while (s[n]) n = n + 1;
  return n;
}

/* The handle of the first held capability of interface `type_id`, or -1 (`cap.self` reflection:
 * authority-neutral, it only re-surfaces what this domain was granted). */
static inline int vm_cap_of(int type_id) {
  int n = __vm_cap_count();
  for (int i = 0; i < n; i = i + 1) {
    int t = 0;
    int h = __vm_cap_at(i, &t);
    if (t == type_id) return h;
  }
  return -1;
}

static int vm_inst_ = -1;
static inline int vm_instantiator_(void) {
  if (vm_inst_ < 0) vm_inst_ = vm_cap_of(6); /* Instantiator = interface 6 */
  return vm_inst_;
}

/* vm_spawn(module, entry, off, size_log2, quota, grants, n, scratch) -> child handle | -errno.
 * `module`: a granted `Module` handle, or -1 for this program. `off`/`size_log2`: the carve, a
 * `1 << size_log2`-aligned sub-range of this window at least the child's declared window.
 * `quota`: raw fuel (0 = the parent's). The child starts immediately; `vm_join` collects it. */
static inline long vm_spawn(long module, long entry, long off, long size_log2, long quota,
                            vm_grant *grants, long n, void *scratch) {
  int *w = (int *)scratch;
  long *q = (long *)scratch;
  w[0] = 0;                /* version */
  w[1] = (int)entry;       /* @4 */
  q[1] = off;              /* @8 */
  w[4] = (int)size_log2;   /* @16 */
  w[5] = -1;               /* @20 pager: u32::MAX = none */
  w[6] = (int)module;      /* @24 */
  w[7] = 0;                /* @28 budget: none — quota-funded */
  q[4] = quota;            /* @32 */
  int *g = (int *)((char *)scratch + 56);
  for (long i = 0; i < n; i = i + 1) {
    g[i * 4 + 0] = (int)(long)grants[i].name;
    g[i * 4 + 1] = (int)vm_strlen_(grants[i].name);
    g[i * 4 + 2] = grants[i].handle;
    g[i * 4 + 3] = 0; /* flags: reserved */
  }
  q[5] = (long)g;          /* @40 */
  q[6] = n;                /* @48 */
  return __vm_instantiate_rec(vm_instantiator_(), (long)scratch);
}

/* vm_join(child) -> the child's entry result (its `main` return), or -errno. */
static inline long vm_join(long child) {
  return __vm_instantiate_join(vm_instantiator_(), child);
}
