// Minimal POSIX <semaphore.h> for the Temen sandbox target — a counting semaphore over the same
// 32-bit atomics + `wait`/`notify` futex the <pthread.h> mutex/cond/barrier are built on
// (DESIGN §12/D56: the VM ships primitives, the libc is policy). Unnamed, in-process semaphores
// only (`pshared` is meaningless in a single shared window and is ignored); `sem_open`/named
// semaphores and `sem_timedwait` are out of scope until programs demand them.
#ifndef __TEMEN_SEMAPHORE_H
#define __TEMEN_SEMAPHORE_H

#include <pthread.h> // the __vm_* primitive declarations

// `__count` is the semaphore value: `sem_wait` CAS-decrements it while positive and parks on the
// word while it is exhausted; `sem_post` increments and wakes one parked waiter. `__vm_wait32`
// re-checks the word atomically, so a post landing between the exhausted read and the park falls
// through instead of losing the wakeup (the same shape as the mutex).
typedef struct {
  int __count;
} sem_t;

static int sem_init(sem_t *s, int pshared, unsigned value) {
  (void)pshared;
  s->__count = (int)value;
  return 0;
}
static int sem_destroy(sem_t *s) {
  (void)s;
  return 0;
}
static int sem_wait(sem_t *s) {
  for (;;) {
    int c = __vm_atomic_load32(&s->__count);
    if (c > 0) {
      if (__vm_atomic_cas32(&s->__count, c, c - 1) == c)
        return 0; // won the decrement
    } else {
      __vm_wait32(&s->__count, c, -1L); // park while exhausted (re-checked atomically)
    }
  }
}
static int sem_trywait(sem_t *s) {
  for (;;) {
    int c = __vm_atomic_load32(&s->__count);
    if (c <= 0)
      return 11; // EAGAIN — POSIX returns -1/errno; the sandbox libc returns the code directly
    if (__vm_atomic_cas32(&s->__count, c, c - 1) == c)
      return 0;
  }
}
static int sem_post(sem_t *s) {
  __vm_atomic_add32(&s->__count, 1);
  __vm_notify(&s->__count, 1);
  return 0;
}
static int sem_getvalue(sem_t *s, int *sval) {
  *sval = __vm_atomic_load32(&s->__count);
  return 0;
}

#endif // __TEMEN_SEMAPHORE_H
