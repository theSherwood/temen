/* spawn.c — the §14 spawn as a C call (#1509): fill the `Instantiator.instantiate_rec` (op 17)
 * config record and its named-grant list, spawn, join.
 *
 * Per-child attenuation is the grant list: a parent narrows handles in its own table first
 * (`AddressSpace.sub`, `Budget.split`, an exported wrapper) and lists a different subset for each
 * child — the child's powerbox is exactly what it is handed here (DESIGN.md §3c "Attenuation needs
 * no new IR"). This file only removes the byte-laying: the 88-byte record (`temen_ir::SpawnRec`) and
 * the `{name_off, name_len, handle, flags}` grant records that every C consumer used to write by hand.
 *
 * Reaches the Instantiator through two frontend builtins (`__vm_instantiate_rec` = op 17,
 * `__vm_instantiate_join` = op 1 — static `call.cap`s like `__vm_exec_module`, since an executor op
 * reaches its seam only through a static `call.cap`) on the handle the guest discovers itself via
 * `cap.self` reflection — no new IR op.
 *
 * Record contract (little-endian, window-relative pointers; fails closed on any other layout) — the
 * v1 **detached** record (#1863): the child gets a window of its own, funded by a `Budget`:
 *   { version: u32 = 1, entry: u32 = 0, off: u64 = 0 (reserved), size_log2: u32, pager: u32 = MAX,
 *     module: i32 (a granted Module), budget: i32 (Budget handle), quota: i64,
 *     grants_ptr: u64, grants_n: u64, args_ptr: u64, args_len: u64,
 *     region: i32 = -1 (none), reserved: u32 = 0, child_off: u64 = 0 }
 * followed here by `n` × 16-byte grant records. `scratch` must be 8-byte aligned and hold
 * `88 + 16 * n` bytes; the names are referenced in place (window pointers), not copied.
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
 * one fails the whole spawn closed.
 *
 * The child binds its imports strictly, so one no grant satisfies refuses the spawn. `VM_EMPTY` as
 * the handle grants nothing: the child's import `name` binds empty and faults if called, a part the
 * parent leaves out on purpose. `name` may also be `*` (every import nothing else satisfies) or a
 * prefix ending in `*`, and only an empty grant may use one (#2219). A child built against the
 * playground libc that uses stdio imports `vm_fs` and `stderr`: grant them, or leave them empty. */
#ifndef VM_EMPTY
#define VM_EMPTY (-0x40000000) /* 0xC000_0000, temen_interp::GRANT_EMPTY */
#endif
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

static int vm_budget_ = -1;
static inline int vm_budget_of_(void) {
  if (vm_budget_ < 0) vm_budget_ = vm_cap_of(14); /* Budget = interface 14 */
  return vm_budget_;
}

/* vm_spawn(module, size_log2, grants, n, args, args_len, scratch) -> child | -errno.
 * `module`: a `Module` handle the host granted you — another program, built to run as a child — and
 * the child starts at its function 0 (#2219). It runs in a window of its own — its module's declared
 * memory, spent from this domain's `Budget`. `size_log2` 0 asks for exactly that; any other value
 * must equal it (else the spawn refuses, -EINVAL).
 * `args`/`args_len`: the spawn-time args payload, copied to the child's args buffer before it starts
 * (the §3e `{argc, envc}` + packed strings a `main(argc, argv)` reads); `args_len` 0 = none.
 * The child's fuel comes from the same `Budget` (the record's per-spawn `quota` is retired, #1944).
 * The child starts immediately; `vm_join` collects it. */
static inline long vm_spawn(long module, long size_log2, vm_grant *grants, long n, void *args,
                            long args_len, void *scratch) {
  int *w = (int *)scratch;
  long *q = (long *)scratch;
  w[0] = 1;                /* version: the detached record */
  w[1] = 0;                /* @4 entry: the child's function 0 */
  q[1] = 0;                /* @8 off: reserved */
  w[4] = (int)size_log2;   /* @16 */
  w[5] = -1;               /* @20 pager: u32::MAX = none */
  w[6] = (int)module;      /* @24 */
  w[7] = vm_budget_of_();  /* @28 the Budget the window spends */
  q[4] = 0;                /* @32 quota: retired, must be 0 */
  int *g = (int *)((char *)scratch + 88);
  for (long i = 0; i < n; i = i + 1) {
    g[i * 4 + 0] = (int)(long)grants[i].name;
    g[i * 4 + 1] = (int)vm_strlen_(grants[i].name);
    g[i * 4 + 2] = grants[i].handle;
    g[i * 4 + 3] = 0; /* flags: reserved */
  }
  q[5] = (long)g;          /* @40 */
  q[6] = n;                /* @48 */
  q[7] = (long)args;       /* @56 */
  q[8] = args_len;         /* @64 */
  w[18] = -1;              /* @72 region: none */
  w[19] = 0;               /* @76 reserved */
  q[10] = 0;               /* @80 child_off */
  return __vm_instantiate_rec(vm_instantiator_(), (long)scratch);
}

/* vm_join(child) -> the child's entry result (its `main` return), or -errno. */
static inline long vm_join(long child) {
  return __vm_instantiate_join(vm_instantiator_(), child);
}
