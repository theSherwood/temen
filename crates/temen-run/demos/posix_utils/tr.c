/* tr(1) — translate or delete bytes from stdin. Supports `tr SET1 SET2` (1:1 byte
 * translation; SET2's last byte repeats when it is shorter) and `tr -d SET1` (delete),
 * with the C escapes `\n \t \r \\ \0` and `a-z` ranges. The everyday shell uses:
 * `tr ' ' '\n'`, `tr 'a-z' 'A-Z'`, `tr -d '\r'`. Byte-at-a-time like `head` — small, honest. */
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);
int u_streq(char *a, char *b);

/* Expand a SET string (escapes + `a-z` ranges) into out[] as bytes; return the count. */
static long tr_expand(char *s, int *out) {
  long n = 0;
  while (*s && n < 256) {
    int c;
    if (*s == '\\') {
      s = s + 1;
      if (*s == 'n') c = '\n';
      else if (*s == 't') c = '\t';
      else if (*s == 'r') c = '\r';
      else if (*s == '0') c = 0;
      else c = (int)(unsigned char)*s; /* \\ and any other: the literal next byte */
      s = s + 1;
    } else {
      c = (int)(unsigned char)*s;
      s = s + 1;
    }
    if (*s == '-' && s[1]) {
      /* a range `c-hi` */
      int hi = (int)(unsigned char)s[1];
      s = s + 2;
      int lo = c;
      while (lo <= hi && n < 256) {
        out[n] = lo;
        n = n + 1;
        lo = lo + 1;
      }
    } else {
      out[n] = c;
      n = n + 1;
    }
  }
  return n;
}

static char tr_in_[4096];
int main(int argc, char **argv) {
  int del = 0;
  int ai = 1;
  if (argc > 1 && u_streq(argv[1], "-d")) {
    del = 1;
    ai = 2;
  }
  int set1[256];
  int set2[256];
  long n1 = 0;
  long n2 = 0;
  if (ai < argc) n1 = tr_expand(argv[ai], set1);
  if (!del && ai + 1 < argc) n2 = tr_expand(argv[ai + 1], set2);
  /* map[b] = the output byte for input byte b (identity by default), or -1 to delete. */
  int map[256];
  int i;
  for (i = 0; i < 256; i = i + 1) map[i] = i;
  for (i = 0; i < n1; i = i + 1) {
    if (del) {
      map[set1[i]] = -1;
    } else if (i < n2) {
      map[set1[i]] = set2[i];
    } else if (n2 > 0) {
      map[set1[i]] = set2[n2 - 1];
    }
  }
  for (;;) {
    long r = read(0, tr_in_, 4096);
    if (r <= 0) break;
    long j;
    for (j = 0; j < r; j = j + 1) {
      int m = map[(int)(unsigned char)tr_in_[j]];
      if (m < 0) continue;
      char o = (char)m;
      if (write(1, &o, 1) != 1) return 1;
    }
  }
  return 0;
}
