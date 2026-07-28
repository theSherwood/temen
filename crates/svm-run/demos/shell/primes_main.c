/* A demo **external command** for the Stage-0 shell (STAGE1.md §5): `primes N` prints every prime
   ≤ N, one per line. Unlike a builtin, this is a *separate compiled-C program* — the shell looks the
   name up in its PATH registry and runs it as an op-13 §14 child (`spawn_cmd`), delivering `argv` and
   re-granting its stdout `Stream`; `write(1, …)` (a chibicc Stream builtin over the granted handle)
   lands in the shell's sink. It reads no stdin (spawn_cmd grants only stdout), so it is a generator,
   not a filter — a small piece of real computation the shell itself cannot do. */
long write(long fd, void *buf, long n);

static void putn(long v) {
  char b[24];
  int i = 24;
  if (v == 0) {
    char z = '0';
    write(1, &z, 1);
    return;
  }
  while (v > 0) {
    b[--i] = (char)('0' + (v % 10));
    v /= 10;
  }
  write(1, b + i, 24 - i);
}

static long atoi_(char *s) {
  long v = 0;
  int i = 0;
  while (s[i] >= '0' && s[i] <= '9') {
    v = v * 10 + (s[i] - '0');
    i++;
  }
  return v;
}

int main(int argc, char **argv) {
  long n = argc > 1 ? atoi_(argv[1]) : 0;
  for (long k = 2; k <= n; k++) {
    int prime = 1;
    for (long d = 2; d * d <= k; d++) {
      if (k % d == 0) {
        prime = 0;
        break;
      }
    }
    if (prime) {
      putn(k);
      write(1, "\n", 1);
    }
  }
  return 0;
}
