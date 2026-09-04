/* mkdir(1) — create each directory operand. `-p` creates missing parents and treats an existing
 * target as success (walking every path prefix and calling mkdir on each, ignoring EEXIST). Without
 * `-p`, a missing parent or existing target is an error. Directories live in the memfs as explicit
 * markers (`__px_mkdir`); the root always exists. */
long __px_mkdir(int cap, long path, long len, long mode);
long u_strlen(char *s);
int u_streq(char *a, char *b);

static char tmp[4096];

int main(int argc, char **argv) {
  int pflag = 0;
  int i = 1;
  while (i < argc && argv[i][0] == '-' && argv[i][1]) {
    if (u_streq(argv[i], "-p")) pflag = 1;
    else break;
    i = i + 1;
  }
  int rc = 0;
  for (; i < argc; i = i + 1) {
    char *path = argv[i];
    long len = u_strlen(path);
    if (!pflag) {
      if (__px_mkdir(0, (long)path, len, 0755) < 0) rc = 1;
      continue;
    }
    /* -p: create every prefix component, ignoring already-exists results */
    long j;
    for (j = 1; j <= len; j = j + 1) {
      if (j == len || path[j] == '/') {
        long k;
        for (k = 0; k < j && k < 4096; k = k + 1) tmp[k] = path[k];
        __px_mkdir(0, (long)tmp, j, 0755);
      }
    }
  }
  return rc;
}
