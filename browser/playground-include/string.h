#ifndef __STRING_H
#define __STRING_H

// Pure-computation <string.h> for the playground (no authority) — all guest C, compiled in on use.
typedef unsigned long size_t;

#include <__pg_linkage.h>

// ---- prototypes (a program unit, #1392) --------------------------------------------------
// Same split as <stdio.h>: a translation unit compiled decls-only (`-include __pg_decls_only.h`)
// sees these prototypes and links against the prebuilt libc unit that carries the bodies once,
// instead of recompiling `strlen`/`memcpy`/`strtok` into every program. The bodies are compiled in
// by default, where `__PG_FN` makes them `static inline` so an unused one is dead-stripped — hence
// the guard, which would otherwise make every one of them a root.
#ifdef __PG_LIBC_DECLS_ONLY
size_t strlen(const char *s);
int strcmp(const char *a, const char *b);
int strncmp(const char *a, const char *b, size_t n);
char *strcpy(char *d, const char *s);
char *strncpy(char *d, const char *s, size_t n);
char *strcat(char *d, const char *s);
char *strchr(const char *s, int c);
char *strrchr(const char *s, int c);
char *strstr(const char *hay, const char *needle);
void *memcpy(void *d, const void *s, size_t n);
void *memmove(void *d, const void *s, size_t n);
void *memset(void *d, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
void *memchr(const void *s, int c, size_t n);
char *strncat(char *d, const char *s, size_t n);
size_t strspn(const char *s, const char *set);
size_t strcspn(const char *s, const char *set);
char *strpbrk(const char *s, const char *set);
char *strtok(char *s, const char *delim);
int __pg_lower(int c);
int strcasecmp(const char *a, const char *b);
int strncasecmp(const char *a, const char *b, size_t n);
char *strerror(int e);
char *strdup(const char *s);
char *strndup(const char *s, size_t n);
#endif /* __PG_LIBC_DECLS_ONLY */

// ---- bodies -----------------------------------------------------------------------------
// In their own file, not behind an `#ifdef` here: chibicc tokenizes a header in full before the
// preprocessor drops the skipped groups, so text left in place would still be *tokenized* by a
// decls-only compile. A separate file is never opened at all.
#ifndef __PG_LIBC_DECLS_ONLY
#include <__pg_string_impl.h>
#endif

#endif
