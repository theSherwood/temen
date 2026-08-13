#ifndef __STDATOMIC_H
#define __STDATOMIC_H

// <stdatomic.h> for the playground — **real** atomics over the VM's atomic builtins (unlike the
// frontend's display-only header that maps them to plain `*p += v`). Threading lessons that rely on
// lock-free atomicity (a shared counter bumped from several vCPUs with no mutex) need the genuine
// article: each `atomic_fetch_add` is a single indivisible RMW op, so the count is correct under
// any interleaving — the whole point of the "atomic counter survives chaos mode" demo.
//
// `_Generic` on the pointee width picks the 32-bit builtin for `int`-sized atomics and the 64-bit
// one for `long`/pointer-sized. Only these two widths are supported (the teaching set); a `short`
// or `char` atomic would fall to the 32/64-bit path and touch neighbouring bytes, so it's excluded.

// The VM atomic builtins (§3e/§4), lowered to the `i32`/`i64` atomic RMW ops. `add` returns the
// *old* value (C11 `atomic_fetch_add` semantics); `cas` returns the old value too.
int __vm_atomic_load32(void *p);
void __vm_atomic_store32(void *p, int v);
int __vm_atomic_add32(void *p, int v);
int __vm_atomic_cas32(void *p, int expected, int desired);
long __vm_atomic_load(void *p);
void __vm_atomic_store(void *p, long v);
long __vm_atomic_add(void *p, long v);

#define ATOMIC_BOOL_LOCK_FREE 1
#define ATOMIC_CHAR_LOCK_FREE 1
#define ATOMIC_SHORT_LOCK_FREE 1
#define ATOMIC_INT_LOCK_FREE 1
#define ATOMIC_LONG_LOCK_FREE 1
#define ATOMIC_LLONG_LOCK_FREE 1
#define ATOMIC_POINTER_LOCK_FREE 1

typedef enum {
  memory_order_relaxed,
  memory_order_consume,
  memory_order_acquire,
  memory_order_release,
  memory_order_acq_rel,
  memory_order_seq_cst,
} memory_order;

// Width dispatch: 8-byte pointees use the i64 builtins, everything else the i32 ones.
#define __pg_atomic_load(obj) \
  _Generic(*(obj), \
    long: __vm_atomic_load, unsigned long: __vm_atomic_load, \
    long long: __vm_atomic_load, unsigned long long: __vm_atomic_load, \
    default: __vm_atomic_load32)((void *)(obj))
#define __pg_atomic_store(obj, val) \
  _Generic(*(obj), \
    long: __vm_atomic_store, unsigned long: __vm_atomic_store, \
    long long: __vm_atomic_store, unsigned long long: __vm_atomic_store, \
    default: __vm_atomic_store32)((void *)(obj), (val))
#define __pg_atomic_add(obj, val) \
  _Generic(*(obj), \
    long: __vm_atomic_add, unsigned long: __vm_atomic_add, \
    long long: __vm_atomic_add, unsigned long long: __vm_atomic_add, \
    default: __vm_atomic_add32)((void *)(obj), (val))

#define atomic_init(addr, val) __pg_atomic_store((addr), (val))
#define atomic_thread_fence(order) ((void)0)
#define atomic_signal_fence(order) ((void)0)
#define atomic_is_lock_free(x) 1

#define atomic_load(addr) __pg_atomic_load(addr)
#define atomic_store(addr, val) __pg_atomic_store((addr), (val))
#define atomic_load_explicit(addr, order) __pg_atomic_load(addr)
#define atomic_store_explicit(addr, val, order) __pg_atomic_store((addr), (val))

#define atomic_fetch_add(obj, val) __pg_atomic_add((obj), (val))
#define atomic_fetch_sub(obj, val) __pg_atomic_add((obj), -(val))
#define atomic_fetch_add_explicit(obj, val, order) __pg_atomic_add((obj), (val))
#define atomic_fetch_sub_explicit(obj, val, order) __pg_atomic_add((obj), -(val))

// Compare-exchange over the 32-bit CAS (int-sized atomics — the teaching case). Returns whether the
// swap happened; on failure writes the current value back through `expected`, as C11 requires. A
// small helper (not a GCC statement-expression) keeps this within the browser IR-codegen C subset.
static inline int __pg_atomic_cas_bool(void *p, int *expected, int desired) {
  int e = *expected;
  int old = __vm_atomic_cas32(p, e, desired);
  if (old == e) return 1;
  *expected = old;
  return 0;
}
#define atomic_compare_exchange_strong(p, expected, desired) \
  __pg_atomic_cas_bool((void *)(p), (int *)(expected), (desired))
#define atomic_compare_exchange_weak(p, expected, desired) \
  atomic_compare_exchange_strong((p), (expected), (desired))
#define atomic_compare_exchange_strong_explicit(p, e, d, s, f) \
  atomic_compare_exchange_strong((p), (e), (d))
#define atomic_compare_exchange_weak_explicit(p, e, d, s, f) \
  atomic_compare_exchange_strong((p), (e), (d))

typedef _Atomic _Bool atomic_bool;
typedef _Atomic char atomic_char;
typedef _Atomic signed char atomic_schar;
typedef _Atomic unsigned char atomic_uchar;
typedef _Atomic short atomic_short;
typedef _Atomic unsigned short atomic_ushort;
typedef _Atomic int atomic_int;
typedef _Atomic unsigned int atomic_uint;
typedef _Atomic long atomic_long;
typedef _Atomic unsigned long atomic_ulong;
typedef _Atomic long long atomic_llong;
typedef _Atomic unsigned long long atomic_ullong;
typedef _Atomic unsigned long atomic_size_t;
typedef _Atomic long atomic_intptr_t;
typedef _Atomic unsigned long atomic_uintptr_t;

#endif
