/* uxnasm — a small Uxntal assembler (a native build-time tool; NOT part of the guest). It covers the
 * uxnasm dialect the demo ROM uses: opcodes with `2`/`r`/`k` modes, `#xx`/`#xxxx` literals, raw hex,
 * `|` and `$` padding, `@label` / `&sublabel` scopes, the reference sigils `. , ; - _ = ! ?` plus bare
 * words (JSI calls), `{ }` / `?{ }` lambdas, `%macro { … }`, `"strings`, `( comments )`, `[ ]`.
 * It follows the reference assembler's encoding choices byte-for-byte on that subset (cross-checked
 * against uxn5's assembler during development); `~include` is not supported.
 *
 *   cc -O2 -o uxnasm uxnasm.c && ./uxnasm demo.tal demo.rom */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LABELS 1024
#define MAX_REFS 4096
#define MAX_MACROS 128
#define MAX_LAMBDAS 64

typedef struct { char name[64]; int addr; } Label;
typedef struct { char name[64]; int addr; char type; } Ref;
typedef struct { char name[64]; char *body; } Macro;

static unsigned char rom[0x10000];
static int ptr = 0x100, top = 0x100; /* write cursor; highest address written + 1 */
static Label labels[MAX_LABELS]; static int nlabels;
static Ref refs[MAX_REFS]; static int nrefs;
static Macro macros[MAX_MACROS]; static int nmacros;
static int lambdas[MAX_LAMBDAS], nlambdas;
static char scope[64] = "on-reset";
static int line = 1;

static const char *ops[32] = {"LIT", "INC", "POP", "NIP", "SWP", "ROT", "DUP", "OVR", "EQU", "NEQ",
  "GTH", "LTH", "JMP", "JCN", "JSR", "STH", "LDZ", "STZ", "LDR", "STR", "LDA", "STA", "DEI", "DEO",
  "ADD", "SUB", "MUL", "DIV", "AND", "ORA", "EOR", "SFT"};

static void die(const char *msg, const char *tok) {
  fprintf(stderr, "uxnasm: line %d: %s: %s\n", line, msg, tok);
  exit(1);
}

static void emit(int b) {
  if (ptr < 0x100 || ptr >= 0x10000) die("write outside the ROM", "");
  rom[ptr++] = (unsigned char)b;
  if (ptr > top) top = ptr;
}
static void emit2(int v) { emit(v >> 8); emit(v & 0xff); }

static int opcode(const char *s) {
  if (!strcmp(s, "BRK")) return 0;
  for (int i = 0; i < 32; i++) {
    if (strncmp(s, ops[i], 3)) continue;
    int op = i ? i : 0x80;
    for (const char *m = s + 3; *m; m++) {
      if (*m == '2') op |= 0x20;
      else if (*m == 'r') op |= 0x40;
      else if (*m == 'k') op |= 0x80;
      else return -1;
    }
    return op;
  }
  return -1;
}

static int ishex(const char *s) {
  size_t n = strlen(s);
  if (n != 2 && n != 4) return 0;
  for (; *s; s++)
    if (!strchr("0123456789abcdefABCDEF", *s)) return 0;
  return 1;
}

/* A label name: `&sub` → "scope/sub", else as written. A name that parses as hex is rejected. */
static void qualify(char *out, const char *name) {
  const char *pre = name[0] == '&' ? scope : "";
  if (name[0] == '&') name++;
  if (strlen(pre) + strlen(name) + 2 > 64) die("name too long", name);
  sprintf(out, "%s%s%s", pre, *pre ? "/" : "", name);
  if (ishex(out)) die("label is a number", out);
}
static void add_label(const char *qualified) {
  if (nlabels == MAX_LABELS) die("too many labels", qualified);
  for (int i = 0; i < nlabels; i++)
    if (!strcmp(labels[i].name, qualified)) die("duplicate label", qualified);
  snprintf(labels[nlabels].name, 64, "%s", qualified);
  labels[nlabels++].addr = ptr;
}
static void add_ref(const char *name, char type, int addr) {
  if (nrefs == MAX_REFS) die("too many references", name);
  qualify(refs[nrefs].name, name);
  refs[nrefs].type = type;
  refs[nrefs++].addr = addr;
}
static int find_label(const char *name) {
  for (int i = 0; i < nlabels; i++)
    if (!strcmp(labels[i].name, name)) return labels[i].addr;
  return -1;
}
static Macro *find_macro(const char *name) {
  for (int i = 0; i < nmacros; i++)
    if (!strcmp(macros[i].name, name)) return &macros[i];
  return NULL;
}

/* Tokenizer over a string cursor; `( … )` comments nest and are skipped. Returns 0 at end. */
static int next_token(const char **src, char *tok, int cap) {
  const char *p = *src;
  int n = 0;
  for (;;) {
    while (*p && strchr(" \t\r\n", *p)) { if (*p == '\n') line++; p++; }
    if (!*p) { *src = p; return 0; }
    if (*p == '(') {
      int depth = 0;
      do { if (*p == '(') depth++; else if (*p == ')') depth--; else if (*p == '\n') line++; p++; } while (*p && depth);
      continue;
    }
    while (*p && !strchr(" \t\r\n", *p)) {
      if (n < cap - 1) tok[n++] = *p;
      p++;
    }
    tok[n] = 0;
    *src = p;
    return 1;
  }
}

static void assemble(const char *src);

static void token(const char *tok, const char **src) {
  char c = tok[0];
  const char *arg = tok + 1;
  if (!strcmp(tok, "[") || !strcmp(tok, "]")) return;
  if (!strcmp(tok, "{") || !strcmp(tok, "?{")) {
    if (nlambdas == MAX_LAMBDAS) die("lambdas nested too deep", tok);
    emit(tok[0] == '?' ? 0x20 : 0x60); emit2(0);
    lambdas[nlambdas++] = ptr - 2;
    return;
  }
  if (!strcmp(tok, "}")) {
    if (!nlambdas) die("unexpected lambda end", tok);
    int at = lambdas[--nlambdas], here = ptr;
    ptr = at; emit2((here - at - 2) & 0xffff); ptr = here;
    return;
  }
  switch (c) {
  case '%': { /* %name { body } — the body is the token run up to the matching `}` */
    char t[256];
    if (nmacros == MAX_MACROS) die("too many macros", tok);
    if (!next_token(src, t, sizeof t) || strcmp(t, "{")) die("macro without body", tok);
    const char *start = *src;
    int depth = 1;
    for (;;) {
      const char *before = *src;
      if (!next_token(src, t, sizeof t)) die("unterminated macro", tok);
      if (!strcmp(t, "{") || !strcmp(t, "?{")) depth++;
      else if (!strcmp(t, "}") && !--depth) {
        Macro *m = &macros[nmacros++];
        qualify(m->name, arg);
        m->body = strndup(start, (size_t)(before - start));
        return;
      }
    }
  }
  case '~': die("~include is not supported", tok); return;
  case '#':
    if (strlen(arg) == 2) { emit(0x80); emit((int)strtol(arg, NULL, 16)); }
    else if (strlen(arg) == 4) { emit(0xa0); emit2((int)strtol(arg, NULL, 16)); }
    else die("bad literal", tok);
    if (!ishex(arg)) die("bad literal", tok);
    return;
  case '|': ptr = (int)strtol(arg, NULL, 16); return;
  case '$': ptr += (int)strtol(arg, NULL, 16); return;
  case '@': {
    char q[64]; qualify(q, arg); add_label(q);
    if (strlen(arg) >= sizeof scope) die("name too long", arg);
    strcpy(scope, arg);
    return;
  }
  case '&': { char q[64]; qualify(q, tok); add_label(q); return; }
  case ',': case '.': emit(0x80); emit(0); add_ref(arg, c, ptr - 1); return;
  case ';': emit(0xa0); emit2(0); add_ref(arg, c, ptr - 2); return;
  case '_': case '-': emit(0); add_ref(arg, c, ptr - 1); return;
  case '=': emit2(0); add_ref(arg, c, ptr - 2); return;
  case '!': emit(0x40); emit2(0); add_ref(arg, c, ptr - 2); return;
  case '?': emit(0x20); emit2(0); add_ref(arg, c, ptr - 2); return;
  case '"': for (; *arg; arg++) emit((unsigned char)*arg); return;
  default: break;
  }
  int op = opcode(tok);
  if (op >= 0) { emit(op); return; }
  Macro *m = find_macro(tok);
  if (m) { int saved = line; assemble(m->body); line = saved; return; }
  if (ishex(tok)) {
    int v = (int)strtol(tok, NULL, 16);
    if (strlen(tok) == 2) emit(v); else emit2(v);
    return;
  }
  emit(0x60); emit2(0); add_ref(tok, 0, ptr - 2); /* bare word: a JSI call */
}

static void assemble(const char *src) {
  char tok[256];
  while (next_token(&src, tok, sizeof tok)) token(tok, &src);
}

static void resolve(void) {
  for (int i = 0; i < nrefs; i++) {
    Ref *r = &refs[i];
    int x = find_label(r->name);
    if (x < 0) die("unknown reference", r->name);
    ptr = r->addr;
    switch (r->type) {
    case ',': case '_': emit((x - r->addr - 2) & 0xff); break;
    case '.': case '-': emit(x & 0xff); break;
    case ';': case '=': emit2(x); break;
    default: emit2((x - r->addr - 2) & 0xffff); break; /* '!' '?' and bare JSI */
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 3) { fprintf(stderr, "usage: uxnasm in.tal out.rom\n"); return 2; }
  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]); return 1; }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *src = malloc((size_t)n + 1);
  if (fread(src, 1, (size_t)n, f) != (size_t)n) { perror("read"); return 1; }
  src[n] = 0;
  fclose(f);
  assemble(src);
  if (nlambdas) die("unterminated lambda", "");
  resolve();
  FILE *o = fopen(argv[2], "wb");
  if (!o) { perror(argv[2]); return 1; }
  fwrite(rom + 0x100, 1, (size_t)(top - 0x100), o);
  fclose(o);
  fprintf(stderr, "uxnasm: %s → %s (%d bytes, %d labels)\n", argv[1], argv[2], top - 0x100, nlabels);
  return 0;
}
