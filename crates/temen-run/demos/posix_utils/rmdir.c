/* rmdir(1) — remove each empty directory operand (`__px_rmdir`; a non-empty dir, a file, or a
 * missing path is an error). The file-removing sibling is rm. */
long __px_rmdir(int cap, long path, long len);
long u_strlen(char *s);

int main(int argc, char **argv) {
  int i;
  int rc = 0;
  for (i = 1; i < argc; i = i + 1)
    if (__px_rmdir(0, (long)argv[i], u_strlen(argv[i])) < 0) rc = 1;
  return rc;
}
