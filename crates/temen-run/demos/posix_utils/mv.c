/* mv(1) — rename SRC to DST via the memfs `__px_rename` (which moves a file's bytes, or re-keys a
 * whole directory subtree). Two operands only; no directory-target mode. */
long __px_rename(int cap, long oldp, long oldlen, long newp, long newlen);
long u_strlen(char *s);

int main(int argc, char **argv) {
  if (argc < 3) return 1; /* usage: mv SRC DST */
  if (__px_rename(0, (long)argv[1], u_strlen(argv[1]), (long)argv[2], u_strlen(argv[2])) < 0)
    return 1;
  return 0;
}
