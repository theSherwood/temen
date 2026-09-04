/* basename(1) — strip the directory prefix (and any trailing slashes) from PATH, then optionally a
 * trailing SUFFIX. `basename /a/b/c.txt .txt` → `c`; `basename /` → `/`; `basename ""` → empty.
 * Args-only (no stdin), like the path-munging staples of real scripts. */
long write(long fd, void *buf, long n);
long u_strlen(char *s);

static char out[4096];

int main(int argc, char **argv) {
  if (argc < 2) return 1; /* missing operand */
  char *s = argv[1];
  long orig = u_strlen(s);
  if (orig == 0) { write(1, "\n", 1); return 0; } /* empty operand → empty line */
  long len = orig;
  while (len > 0 && s[len - 1] == '/') len = len - 1;
  if (len == 0) { write(1, "/\n", 2); return 0; } /* all slashes → "/" */
  long start = 0;
  long i;
  for (i = 0; i < len; i = i + 1) if (s[i] == '/') start = i + 1;
  long blen = len - start;
  for (i = 0; i < blen && i < 4096; i = i + 1) out[i] = s[start + i];
  if (argc >= 3) {
    char *suf = argv[2];
    long sl = u_strlen(suf);
    if (sl > 0 && sl < blen) {
      long k;
      long match = 1;
      for (k = 0; k < sl; k = k + 1) if (out[blen - sl + k] != suf[k]) { match = 0; break; }
      if (match) blen = blen - sl;
    }
  }
  write(1, out, blen);
  write(1, "\n", 1);
  return 0;
}
