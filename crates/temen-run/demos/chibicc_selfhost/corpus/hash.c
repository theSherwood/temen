// FNV-1a hash of a fixed string; return the low byte — pointers, char loop, bit ops, unsigned.
unsigned int fnv1a(const char *s) {
  unsigned int h = 2166136261u;
  while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
  return h;
}
int main(void) {
  return (int)(fnv1a("the small trustworthy core") & 0xff);
}
