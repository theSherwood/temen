/*
 * temen.h — C bindings for the Temen embedding runtime (POWERBOX.md Phase 5).
 *
 * Pipeline: parse a module (text or binary IR; a paramless exported _start over an import
 * manifest) -> bind host capabilities by name (built-ins or your own C callbacks) -> temen_instantiate* ->
 * temen_instance_run / temen_instance_run_diff -> read the outcome and captured stdout/stderr.
 *
 * Conventions:
 *   - Handles are opaque pointers; functions that say "consumes" take ownership (do not free after).
 *   - A NULL return or a non-zero status means failure; call temen_last_error() for the message.
 *   - Panics never cross the boundary (they become a NULL/error return).
 *
 * Link against libtemen_capi.a (staticlib) or libtemen_capi.{so,dylib} (cdylib).
 */
#ifndef TEMEN_H
#define TEMEN_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. */
#define TEMEN_OK 0
#define TEMEN_ERR_NULL 1
#define TEMEN_ERR_FAILED 2
#define TEMEN_ERR_PANIC 3

/* Backend selectors for temen_instance_run. */
#define TEMEN_BACKEND_TREEWALK 0
#define TEMEN_BACKEND_BYTECODE 1
#define TEMEN_BACKEND_JIT 2

/* temen_run_outcome_kind values. */
#define TEMEN_OUTCOME_RETURNED 0
#define TEMEN_OUTCOME_EXITED 1

/* Opaque handles. */
typedef struct TemenModule TemenModule;
typedef struct TemenImports TemenImports;
typedef struct TemenInstance TemenInstance;
typedef struct TemenRun TemenRun;
/* The calling guest's linear-memory window, passed to a host-fn callback for that call only. */
typedef struct TemenGuestMem TemenGuestMem;

/* The last error message on this thread (or NULL). Valid until the next temen-capi call. */
const char *temen_last_error(void);

/* ---- Module ---- */
TemenModule *temen_module_parse_text(const char *ir);
TemenModule *temen_module_decode(const uint8_t *bytes, size_t len);
void temen_module_free(TemenModule *m);

/* ---- Imports registry (wasm-style name -> capability) ---- */
TemenImports *temen_imports_new(void);
int32_t temen_imports_provide_stdout(TemenImports *i, const char *name);
int32_t temen_imports_provide_stdin(TemenImports *i, const char *name);
int32_t temen_imports_provide_exit(TemenImports *i, const char *name);
int32_t temen_imports_provide_clock(TemenImports *i, const char *name);

/*
 * A host-capability callback: compute up to results_cap outputs from n_args inputs for operation op.
 * Return the number of results written (>= 0), or a negative value to trap the capability call.
 * ctx is the opaque pointer registered alongside the callback. mem is the calling guest's window
 * (NULL if the module declares none), accessible via temen_guest_read/temen_guest_write for this call
 * only — do not retain it past the callback.
 */
typedef int32_t (*TemenHostFn)(void *ctx, uint32_t op, const int64_t *args, size_t n_args,
                             int64_t *results, size_t results_cap, TemenGuestMem *mem);
int32_t temen_imports_provide_host_fn(TemenImports *i, const char *name, uint32_t op, TemenHostFn fn,
                                    void *ctx);
void temen_imports_free(TemenImports *i);

/*
 * Read/write the guest window from inside a host-fn callback, bounds-checked (fail-closed): each
 * returns TEMEN_OK, or TEMEN_ERR_FAILED (nothing transferred) if mem/buf is NULL or [ptr, ptr+len) is
 * not wholly within the window (and, for write, writable). The same §7 confinement the built-ins get.
 */
int32_t temen_guest_read(const TemenGuestMem *mem, uint64_t ptr, uint8_t *dst, size_t len);
int32_t temen_guest_write(TemenGuestMem *mem, uint64_t ptr, const uint8_t *src, size_t len);

/* ---- Instantiate (consume the module / imports) ---- */
TemenInstance *temen_instantiate(TemenModule *m);                          /* fixed §3e powerbox */
TemenInstance *temen_instantiate_with_imports(TemenModule *m, TemenImports *imports); /* by name */
void temen_instance_free(TemenInstance *i);

/*
 * ---- Memory-access hooks ----
 * Opt an instance into observing every guest memory access. TemenMemEvent.kind is one of the
 * TEMEN_MEM_* constants; for LOAD/STORE/ATOMIC_* addr is the effective guest address and size the
 * access width in bytes (v128 is LOAD/STORE, size 16); for COPY addr=dst, src=src, size=len; for
 * FILL addr=dst, size=len (src=0). The hook returns 0 to allow the access, non-zero to veto it (the
 * run aborts with a capability trap). The ev pointer is valid only for the callback.
 */
enum {
  TEMEN_MEM_LOAD = 0,
  TEMEN_MEM_STORE = 1,
  TEMEN_MEM_ATOMIC_LOAD = 2,
  TEMEN_MEM_ATOMIC_STORE = 3,
  TEMEN_MEM_ATOMIC_RMW = 4,
  TEMEN_MEM_ATOMIC_CMPXCHG = 5,
  TEMEN_MEM_COPY = 6,
  TEMEN_MEM_FILL = 7
};
typedef struct {
  int32_t kind;  /* TEMEN_MEM_* */
  uint64_t addr; /* scalar/atomic: effective addr; COPY/FILL: dst */
  uint64_t src;  /* COPY: src; else 0 */
  uint64_t size; /* scalar/atomic: width in bytes; COPY/FILL: len in bytes */
} TemenMemEvent;
typedef int32_t (*TemenMemHook)(void *ctx, const TemenMemEvent *ev);
/* Consumes `i`; returns a new hooked instance (run on any backend), or NULL on failure. Give a
 * hooked run more fuel than the pristine module (it executes more instructions). */
TemenInstance *temen_instance_with_mem_hooks(TemenInstance *i, TemenMemHook hook, void *ctx);

/* ---- Run config ---- (a NULL pointer means all defaults; *_set flags select a field). */
typedef struct {
  uint64_t fuel;       /* per-op budget for the interpreters (if fuel_set); ignored by the JIT */
  int32_t fuel_set;
  uint64_t deadline_ms; /* JIT detect-and-kill deadline (if deadline_set); ignored by interps */
  int32_t deadline_set;
  size_t max_fibers; /* §15 spawn quota (0 = default) */
  size_t max_vcpus;  /* §15 vCPU cap / "CPUs available" (0 = default) */
  const uint8_t *stdin_bytes; /* guest stdin (NULL/0 = empty) */
  size_t stdin_len;
  uint8_t memory_size_log2; /* window override (if memory_set) */
  int32_t memory_set;
} TemenRunConfig;

/* ---- Run ---- */
TemenRun *temen_instance_run(TemenInstance *i, int32_t backend, const TemenRunConfig *config);
TemenRun *temen_instance_run_diff(TemenInstance *i, const TemenRunConfig *config);

/* ---- Reactor sessions (Phase 6): instantiate once, call exports repeatedly, state persists ---- */
typedef struct TemenSession TemenSession;
TemenSession *temen_instance_start(const TemenInstance *i, int32_t backend, const TemenRunConfig *config);
/* Call `name` with n_args i64 args; write up to results_cap i64 results + *n_results. 0 = TEMEN_OK. */
int32_t temen_session_call_export(TemenSession *s, const char *name, const int64_t *args, size_t n_args,
                                int64_t *results, size_t results_cap, size_t *n_results);
const uint8_t *temen_session_stdout(const TemenSession *s, size_t *len);
void temen_session_free(TemenSession *s);

/* ---- Run results ---- (pointers valid until temen_run_free). */
const uint8_t *temen_run_stdout(const TemenRun *r, size_t *len);
const uint8_t *temen_run_stderr(const TemenRun *r, size_t *len);
int32_t temen_run_outcome_kind(const TemenRun *r);
int32_t temen_run_exit_code(const TemenRun *r);
size_t temen_run_result_count(const TemenRun *r);
int64_t temen_run_result(const TemenRun *r, size_t idx);
void temen_run_free(TemenRun *r);

#ifdef __cplusplus
}
#endif

#endif /* TEMEN_H */
