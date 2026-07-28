#ifndef __STDARG_H
#define __STDARG_H

// SVM flat-buffer varargs ABI (DESIGN.md §3d), matching frontend/chibicc/include/stdarg.h: the
// caller marshals each promoted variadic argument into one 8-byte slot in a contiguous buffer and
// passes a pointer to it as a hidden trailing argument — `__va_area__` inside the callee.
typedef char *va_list[1];

#define va_start(ap, last) ((ap)[0] = __va_area__)
#define va_end(ap) ((void)0)
#define va_copy(dest, src) ((dest)[0] = (src)[0])
#define va_arg(ap, ty) (*(ty *)(((ap)[0] += 8) - 8))

#define __GNUC_VA_LIST 1
typedef va_list __gnuc_va_list;

#endif
