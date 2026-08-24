/* Demo consumer for the increment-1 emit-object libc (`mini_libc.c`, task #20). A separate TU that
 * uses the libc's `malloc`, `strcpy`/`strlen`, and varargs `printf` — every call crosses the unit
 * boundary and resolves at link. `main` builds a malloc'd array on the (linked) data heap, copies a
 * string, and prints a deterministic line, so the linked program's stdout is a fixed oracle checked
 * on interp==JIT under the powerbox. Returns the array sum (140) as the exit value.
 */

typedef unsigned long size_t;

extern void *malloc(size_t);
extern size_t strlen(const char *);
extern char *strcpy(char *, const char *);
extern int printf(const char *, ...);

int main(void) {
  int *a = (int *)malloc(8 * sizeof(int));
  int s = 0;
  for (int i = 0; i < 8; i++) {
    a[i] = i * i;
    s += a[i];
  }
  char buf[32];
  strcpy(buf, "hello");
  printf("sum=%d first=%d last=%d len=%d str=%s hex=%x char=%c\n",
         s, a[0], a[7], (int)strlen(buf), buf, 255, '!');
  return s;
}
