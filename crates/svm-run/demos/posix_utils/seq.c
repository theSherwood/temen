/* seq(1) — `seq LAST` or `seq FIRST LAST`, one integer per line. */
long write(long fd, void *buf, long n);
long u_atoi(char *s);
long u_putn(long fd, long v);

int main(int argc, char **argv) {
  long first = 1, last;
  if (argc == 2) last = u_atoi(argv[1]);
  else if (argc == 3) { first = u_atoi(argv[1]); last = u_atoi(argv[2]); }
  else return 1;
  long v;
  for (v = first; v <= last; v = v + 1) {
    if (u_putn(1, v) < 0) return 1;
    if (write(1, "\n", 1) != 1) return 1;
  }
  return 0;
}
