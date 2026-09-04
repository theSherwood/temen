/* dirname(1) — the directory prefix of PATH (POSIX): strip trailing slashes, drop the last
 * component, strip the slashes before it. `dirname /a/b/c` → `/a/b`; `dirname /a` → `/`;
 * `dirname a` → `.`; `dirname /` → `/`; `dirname ""` → `.`. Args-only, no stdin. */
long write(long fd, void *buf, long n);
long u_strlen(char *s);

int main(int argc, char **argv) {
  if (argc < 2) return 1; /* missing operand */
  char *s = argv[1];
  long orig = u_strlen(s);
  long len = orig;
  while (len > 0 && s[len - 1] == '/') len = len - 1;
  if (len == 0) {
    if (orig == 0) { write(1, ".\n", 2); return 0; } /* empty operand */
    write(1, "/\n", 2);                              /* all slashes → "/" */
    return 0;
  }
  long lastslash = -1;
  long i;
  for (i = 0; i < len; i = i + 1) if (s[i] == '/') lastslash = i;
  if (lastslash < 0) { write(1, ".\n", 2); return 0; } /* no directory part */
  long dl = lastslash;
  while (dl > 0 && s[dl - 1] == '/') dl = dl - 1;
  if (dl == 0) { write(1, "/\n", 2); return 0; } /* directory is the root */
  write(1, s, dl);
  write(1, "\n", 1);
  return 0;
}
