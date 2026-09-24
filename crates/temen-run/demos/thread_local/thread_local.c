/* `_Thread_local` across threads (#1715): every thread gets its own copy of each thread-local,
 * starting from the program's initial values — never another thread's writes, and never the root's
 * current values. Built by both C compilers (chibicc, and clang through the LLVM on-ramp) against the
 * same <pthread.h>, and must print the same thing on both.
 *
 * The root sets its `counter` to 5 before starting any thread. Worker `me` then adds `me` to its own
 * `counter` 1000 times and counts the calls in a block-scope `static _Thread_local`, and records
 * `counter*100000 + calls*10 + *pshared + me` (`pshared` is initialized to `&shared`, so the block
 * carries a relocated pointer). Workers write disjoint result slots; main prints them after joining,
 * then its own `counter` and `tag`, untouched by the workers. */
#include <pthread.h>

int write(int fd, char *buf, long n);

#define NWORKERS 3

_Thread_local long counter = 100;
_Thread_local char tag[8] = "root";
static long shared = 7;
_Thread_local long *pshared = &shared;

static long results[NWORKERS];

static void *worker(void *arg) {
  long me = (long)arg;
  static _Thread_local int calls;
  for (int i = 0; i < 1000; i++) {
    counter += me;
    calls++;
  }
  tag[0] = 'w';
  tag[1] = (char)('0' + me);
  results[me] = counter * 100000 + calls * 10 + *pshared + (tag[1] - '0');
  return 0;
}

static void print_long(long v) {
  char buf[24];
  char tmp[24];
  int n = 0, t = 0;
  if (v == 0)
    buf[n++] = '0';
  while (v > 0) {
    tmp[t++] = (char)('0' + (v % 10));
    v /= 10;
  }
  while (t > 0)
    buf[n++] = tmp[--t];
  buf[n++] = '\n';
  write(1, buf, n);
}

int main(void) {
  counter = 5;
  pthread_t workers[NWORKERS];
  for (long i = 0; i < NWORKERS; i++)
    pthread_create(&workers[i], 0, worker, (void *)i);
  for (int i = 0; i < NWORKERS; i++)
    pthread_join(workers[i], 0);
  for (int i = 0; i < NWORKERS; i++)
    print_long(results[i]);
  print_long(counter);
  write(1, tag, 4);
  write(1, "\n", 1);
  return 0;
}
