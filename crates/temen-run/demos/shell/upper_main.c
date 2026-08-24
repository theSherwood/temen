/* A demo **filter** external command for the Stage-0 shell (STAGE1.md §5): `upper` reads its stdin,
   uppercases every ASCII letter, and writes the result to stdout. Unlike `primes` (a generator, only
   stdout granted), a filter also gets a `"stdin"` grant — a read-only pipe the shell fills from the
   command's input (a `< file` redirect or a pipe stage). `read(0, …)` / `write(1, …)` are chibicc
   Stream builtins over the granted `"stdin"` / `"stdout"` handles; `read` returns 0 at EOF (the pipe
   drained), ending the loop. It is spawned as an op-13 §14 child, exactly like any external command. */
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);

int main(void) {
  char buf[256];
  long r;
  while ((r = read(0, buf, 256)) > 0) {
    for (long i = 0; i < r; i++) {
      char c = buf[i];
      if (c >= 'a' && c <= 'z') buf[i] = (char)(c - 32);
    }
    write(1, buf, r);
  }
  return 0;
}
