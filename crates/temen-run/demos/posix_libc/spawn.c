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
 * The record is `<temen.h>`'s `struct vm_spawn_rec` (the v1 **detached** record, #1863: the child
 * gets a window of its own, funded by a `Budget`), followed here by `n` × 16-byte grant records.
 * `scratch` must be 8-byte aligned and hold `88 + 16 * n` bytes; the names are referenced in place
 * (window pointers), not copied.
 *
 * Freestanding (no libc headers) for the chibicc harnesses; every helper is `static inline` so the
 * frontend's dead-code pass drops what a program never calls.
 */
#include <temen.h>

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
  struct vm_spawn_rec init = VM_SPAWN_REC_INIT;
  struct vm_spawn_rec *r = (struct vm_spawn_rec *)scratch;
  *r = init;
  r->size_log2 = (unsigned int)size_log2;
  r->module = (int)module;
  r->budget = vm_budget_of_(); /* the Budget the window spends */
  int *g = (int *)((char *)scratch + sizeof(struct vm_spawn_rec));
  for (long i = 0; i < n; i = i + 1) {
    g[i * 4 + 0] = (int)(long)grants[i].name;
    g[i * 4 + 1] = (int)vm_strlen_(grants[i].name);
    g[i * 4 + 2] = grants[i].handle;
    g[i * 4 + 3] = 0; /* flags: reserved */
  }
  r->grants_ptr = (long)g;
  r->grants_n = n;
  r->args_ptr = (long)args;
  r->args_len = args_len;
  return __vm_instantiate_rec(vm_instantiator_(), (long)r);
}

/* vm_join(child, &status) -> the child's entry result (its `main` return), with `status` 0; or 0, with
 * `status` the wire code of the trap the child ended in (`temen_ir::trap_code`: out of fuel, a fault,
 * `kill`, or `exit(n)` with `n` in the high 32 bits). The child's end is never the caller's (#2066),
 * and either way it is reaped. */
static inline long vm_join(long child, long *status) {
  return __vm_instantiate_join(vm_instantiator_(), child, status);
}
