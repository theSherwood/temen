#ifndef __UNISTD_H
#define __UNISTD_H
// <unistd.h> — the file syscalls the powerbox provides (write/read/open/close over the ambient Stream
// / fs cap), declared with the SAME signatures <stdio.h> uses so the two co-include cleanly, plus
// inert stubs for the driver-only calls (fork/exec) so chibicc.h parses. A cc1 --emit-ir compile uses
// none of the process calls.
#include <stddef.h>
typedef long ssize_t;
typedef int pid_t;
// Powerbox syscalls (the on-ramp lowers write/read to the Stream cap; open/close hit the fs cap).
int write(int fd, char *buf, long n);
int read(int fd, char *buf, long n);
int open(const char *path, int flags, ...);
int close(int fd);
long lseek(int fd, long off, int whence);
static inline int unlink(const char *p) { (void)p; return 0; }
static inline int access(const char *p, int m) { (void)p; (void)m; return -1; }
static inline pid_t fork(void) { return -1; }
static inline pid_t getpid(void) { return 1; }
static inline int isatty(int fd) { (void)fd; return 0; }
#endif
