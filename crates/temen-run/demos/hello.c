// The same greeting as hello.temt, but in C — compiled through the chibicc frontend and run
// sandboxed:  temen-run crates/temen-run/demos/hello.c
//
// `write` is a powerbox builtin (the Stream capability, §3e); the frontend lowers it to a
// `cap.call` on the granted stdout handle.

int write(int fd, char *buf, long n);

int main(void) {
  char *msg = "hello, sandbox!\n";
  long n = 0;
  while (msg[n]) n++;
  write(1, msg, n);
  return 0;
}
