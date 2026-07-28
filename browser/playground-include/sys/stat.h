#ifndef __SYS_STAT_H
#define __SYS_STAT_H
// <sys/stat.h> — chibicc's file_exists()/__TIMESTAMP__ stat() the source; in the sandbox files come
// from the memfs, so a stat over the powerbox is not modeled: stat() returns -1 (ENOENT), which
// file_exists() reads as "absent" and __TIMESTAMP__ handles with its "??? " fallback.
#include <sys/types.h>
struct stat {
  dev_t st_dev; ino_t st_ino; mode_t st_mode; unsigned long st_nlink;
  unsigned int st_uid, st_gid; dev_t st_rdev; off_t st_size;
  long st_blksize, st_blocks; time_t st_atime, st_mtime, st_ctime;
};
#define S_IFMT  0170000
#define S_IFREG 0100000
#define S_IFDIR 0040000
#define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
static inline int stat(const char *p, struct stat *s) { (void)p; (void)s; return -1; }
static inline int fstat(int fd, struct stat *s) { (void)fd; (void)s; return -1; }
static inline int mkdir(const char *p, mode_t m) { (void)p; (void)m; return -1; }
#endif
