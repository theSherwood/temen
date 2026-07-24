// Sum 1..100, plus a small array reduction — arithmetic, loops, arrays.
int main(void) {
  int s = 0;
  for (int i = 1; i <= 100; i++) s += i;
  int xs[6] = {5, 3, 9, 1, 7, 2};
  for (int i = 0; i < 6; i++) s += xs[i] * (i + 1);
  return s % 256;
}
