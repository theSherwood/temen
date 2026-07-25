// Bubble-sort a small array; return a position-weighted checksum — nested loops, swaps, mutation.
int main(void) {
  int a[8] = {42, 7, 13, 99, 1, 64, 28, 5};
  int n = 8;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n - 1 - i; j++)
      if (a[j] > a[j + 1]) { int t = a[j]; a[j] = a[j + 1]; a[j + 1] = t; }
  int c = 0;
  for (int i = 0; i < n; i++) c += a[i] * (i + 1);
  return c % 256;
}
