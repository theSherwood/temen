#ifndef __SYS_WAIT_H
#define __SYS_WAIT_H
#include <sys/types.h>
// Driver-only (fork/exec); inert in the sandbox so chibicc.h parses.
static inline pid_t wait(int *s) { (void)s; return -1; }
static inline pid_t waitpid(pid_t p, int *s, int f) { (void)p; (void)s; (void)f; return -1; }
#define WIFEXITED(s) 1
#define WEXITSTATUS(s) ((s) & 0xff)
#endif
