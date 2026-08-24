/* echo(1) — argv joined by single spaces; -n suppresses the trailing newline. */
long write(long fd, void *buf, long n);
long u_strlen(char *s);
int u_streq(char *a, char *b);

int main(int argc, char **argv) {
  int i = 1, nl = 1;
  if (i < argc && u_streq(argv[i], "-n")) { nl = 0; i = i + 1; }
  for (; i < argc; i = i + 1) {
    if (write(1, argv[i], u_strlen(argv[i])) < 0) return 1;
    if (i + 1 < argc && write(1, " ", 1) != 1) return 1;
  }
  if (nl && write(1, "\n", 1) != 1) return 1;
  return 0;
}
