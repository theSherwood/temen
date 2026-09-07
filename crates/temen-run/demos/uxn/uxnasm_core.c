/* uxnasm — the Uxntal assembler core, freestanding (no libc) so it runs both as the native build tool
 * (`uxnasm.c` wraps it with stdio) and inside the Temen guest (`main.c` assembles the served `boot.tal`).
 * The dialect: opcodes with `2`/`r`/`k` modes, `#xx`/`#xxxx` literals, raw hex, `|` and `$` padding,
 * `@label` / `&sublabel` scopes, the reference sigils `. , ; - _ = ! ?` plus bare words (JSI calls),
 * `{ }` / `?{ }` lambdas, `%macro { … }`, `"strings`, `( comments )`, `[ ]`. It follows the reference
 * assembler's encoding byte-for-byte on that subset (cross-checked against uxn5's assembler);
 * `~include` is not supported.
 *
 *   int uxnasm_assemble(const char *src, int len, Uint8 *rom, int *rom_len);
 * assembles into `rom` (0x10000 bytes; the ROM is rom[0x100 .. 0x100 + *rom_len]) and returns 1, or 0
 * on error with `uxnasm_error` (a message) and `uxnasm_error_line` set. */
#include "uxn.h"

#define ASM_NAME 64
#define ASM_LABELS 1024
#define ASM_REFS 4096
#define ASM_MACROS 128
#define ASM_LAMBDAS 64
#define ASM_TOKEN 256

typedef struct { char name[ASM_NAME]; int addr; } AsmLabel;
typedef struct { char name[ASM_NAME]; int addr, line; char type; } AsmRef;
typedef struct { char name[ASM_NAME]; const char *body; int len; } AsmMacro;

static struct {
  Uint8 *rom;
  int ptr, top; /* write cursor; highest address written + 1 */
  AsmLabel labels[ASM_LABELS]; int nlabels;
  AsmRef refs[ASM_REFS]; int nrefs;
  AsmMacro macros[ASM_MACROS]; int nmacros;
  int lambdas[ASM_LAMBDAS], nlambdas;
  char scope[ASM_NAME];
  int line, failed;
} A;

char uxnasm_error[ASM_TOKEN + 64];
int uxnasm_error_line;

static const char *asm_ops[32] = {"LIT", "INC", "POP", "NIP", "SWP", "ROT", "DUP", "OVR", "EQU", "NEQ",
  "GTH", "LTH", "JMP", "JCN", "JSR", "STH", "LDZ", "STZ", "LDR", "STR", "LDA", "STA", "DEI", "DEO",
  "ADD", "SUB", "MUL", "DIV", "AND", "ORA", "EOR", "SFT"};

/* --- tiny string helpers (no libc) ------------------------------------------------------------- */
static int s_len(const char *s) { int n = 0; while (s[n]) n++; return n; }
static int s_eq(const char *a, const char *b) { while (*a && *a == *b) a++, b++; return *a == *b; }
static void s_cpy(char *d, const char *s) { while ((*d++ = *s++)); }
static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}
static int ishex(const char *s) { /* exactly 2 or 4 hex digits */
  int n = s_len(s);
  if (n != 2 && n != 4) return 0;
  for (; *s; s++) if (hexval(*s) < 0) return 0;
  return 1;
}
static int parsehex(const char *s) { int v = 0; for (; *s && hexval(*s) >= 0; s++) v = v * 16 + hexval(*s); return v; }
static int isspace_(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

static void fail(const char *msg, const char *tok) {
  if (A.failed) return;
  A.failed = 1;
  uxnasm_error_line = A.line;
  char *o = uxnasm_error;
  const char *m = msg;
  while (*m) *o++ = *m++;
  *o++ = ':'; *o++ = ' ';
  int n = 0;
  while (*tok && n++ < ASM_TOKEN - 1) *o++ = *tok++;
  *o = 0;
}

static void emit(int b) {
  if (A.ptr < 0x100 || A.ptr >= 0x10000) { fail("write outside the ROM", ""); return; }
  A.rom[A.ptr++] = (Uint8)b;
  if (A.ptr > A.top) A.top = A.ptr;
}
static void emit2(int v) { emit(v >> 8); emit(v & 0xff); }

static int opcode(const char *s) {
  if (s_eq(s, "BRK")) return 0;
  for (int i = 0; i < 32; i++) {
    if (s[0] != asm_ops[i][0] || s[1] != asm_ops[i][1] || s[2] != asm_ops[i][2]) continue;
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

/* A label name: `&sub` → "scope/sub", else as written. A name that parses as hex is rejected. */
static void qualify(char *out, const char *name) {
  const char *pre = "";
  if (name[0] == '&') { pre = A.scope; name++; }
  if (s_len(pre) + s_len(name) + 2 > ASM_NAME) { fail("name too long", name); out[0] = 0; return; }
  char *o = out;
  if (*pre) { for (const char *p = pre; *p; p++) *o++ = *p; *o++ = '/'; }
  s_cpy(o, name);
  if (ishex(out)) fail("label is a number", out);
}
static void add_label(const char *qualified) {
  if (A.nlabels == ASM_LABELS) { fail("too many labels", qualified); return; }
  for (int i = 0; i < A.nlabels; i++)
    if (s_eq(A.labels[i].name, qualified)) { fail("duplicate label", qualified); return; }
  s_cpy(A.labels[A.nlabels].name, qualified);
  A.labels[A.nlabels++].addr = A.ptr;
}
static void add_ref(const char *name, char type, int addr) {
  if (A.nrefs == ASM_REFS) { fail("too many references", name); return; }
  qualify(A.refs[A.nrefs].name, name);
  A.refs[A.nrefs].type = type;
  A.refs[A.nrefs].line = A.line;
  A.refs[A.nrefs++].addr = addr;
}
static int find_label(const char *name) {
  for (int i = 0; i < A.nlabels; i++)
    if (s_eq(A.labels[i].name, name)) return A.labels[i].addr;
  return -1;
}
static AsmMacro *find_macro(const char *name) {
  for (int i = 0; i < A.nmacros; i++)
    if (s_eq(A.macros[i].name, name)) return &A.macros[i];
  return 0;
}

/* Tokenizer over [p, end); `( … )` comments nest and are skipped. Returns 0 at the end. */
static int next_token(const char **src, const char *end, char *tok) {
  const char *p = *src;
  int n = 0;
  for (;;) {
    while (p < end && isspace_(*p)) { if (*p == '\n') A.line++; p++; }
    if (p >= end) { *src = p; return 0; }
    if (*p == '(') {
      int depth = 0;
      do { if (*p == '(') depth++; else if (*p == ')') depth--; else if (*p == '\n') A.line++; p++; } while (p < end && depth);
      continue;
    }
    while (p < end && !isspace_(*p)) {
      if (n < ASM_TOKEN - 1) tok[n++] = *p;
      p++;
    }
    tok[n] = 0;
    *src = p;
    return 1;
  }
}

static void assemble_range(const char *src, const char *end);

static void token(const char *tok, const char **src, const char *end) {
  char c = tok[0];
  const char *arg = tok + 1;
  if (s_eq(tok, "[") || s_eq(tok, "]")) return;
  if (s_eq(tok, "{") || s_eq(tok, "?{")) {
    if (A.nlambdas == ASM_LAMBDAS) { fail("lambdas nested too deep", tok); return; }
    emit(tok[0] == '?' ? 0x20 : 0x60); emit2(0);
    A.lambdas[A.nlambdas++] = A.ptr - 2;
    return;
  }
  if (s_eq(tok, "}")) {
    if (!A.nlambdas) { fail("unexpected lambda end", tok); return; }
    int at = A.lambdas[--A.nlambdas], here = A.ptr;
    A.ptr = at; emit2((here - at - 2) & 0xffff); A.ptr = here;
    return;
  }
  switch (c) {
  case '%': { /* %name { body } — the body is the token run up to the matching `}` */
    char t[ASM_TOKEN];
    if (A.nmacros == ASM_MACROS) { fail("too many macros", tok); return; }
    if (!next_token(src, end, t) || !s_eq(t, "{")) { fail("macro without body", tok); return; }
    const char *start = *src;
    int depth = 1;
    for (;;) {
      const char *before = *src;
      if (!next_token(src, end, t)) { fail("unterminated macro", tok); return; }
      if (s_eq(t, "{") || s_eq(t, "?{")) depth++;
      else if (s_eq(t, "}") && !--depth) {
        AsmMacro *m = &A.macros[A.nmacros++];
        qualify(m->name, arg);
        m->body = start;
        m->len = (int)(before - start);
        return;
      }
    }
  }
  case '~': fail("~include is not supported", tok); return;
  case '#':
    if (!ishex(arg)) { fail("bad literal", tok); return; }
    if (s_len(arg) == 2) { emit(0x80); emit(parsehex(arg)); }
    else { emit(0xa0); emit2(parsehex(arg)); }
    return;
  case '|': A.ptr = parsehex(arg); return;
  case '$': A.ptr += parsehex(arg); return;
  case '@': {
    char q[ASM_NAME]; qualify(q, arg); add_label(q);
    if (s_len(arg) >= ASM_NAME) { fail("name too long", arg); return; }
    s_cpy(A.scope, arg);
    return;
  }
  case '&': { char q[ASM_NAME]; qualify(q, tok); add_label(q); return; }
  case ',': case '.': emit(0x80); emit(0); add_ref(arg, c, A.ptr - 1); return;
  case ';': emit(0xa0); emit2(0); add_ref(arg, c, A.ptr - 2); return;
  case '_': case '-': emit(0); add_ref(arg, c, A.ptr - 1); return;
  case '=': emit2(0); add_ref(arg, c, A.ptr - 2); return;
  case '!': emit(0x40); emit2(0); add_ref(arg, c, A.ptr - 2); return;
  case '?': emit(0x20); emit2(0); add_ref(arg, c, A.ptr - 2); return;
  case '"': for (; *arg; arg++) emit((Uint8)*arg); return;
  default: break;
  }
  int op = opcode(tok);
  if (op >= 0) { emit(op); return; }
  AsmMacro *m = find_macro(tok);
  if (m) { int saved = A.line; assemble_range(m->body, m->body + m->len); A.line = saved; return; }
  if (ishex(tok)) {
    int v = parsehex(tok);
    if (s_len(tok) == 2) emit(v); else emit2(v);
    return;
  }
  emit(0x60); emit2(0); add_ref(tok, 0, A.ptr - 2); /* bare word: a JSI call */
}

static void assemble_range(const char *src, const char *end) {
  char tok[ASM_TOKEN];
  while (!A.failed && next_token(&src, end, tok)) token(tok, &src, end);
}

static void resolve(void) {
  for (int i = 0; i < A.nrefs && !A.failed; i++) {
    AsmRef *r = &A.refs[i];
    int x = find_label(r->name);
    if (x < 0) { A.line = r->line; fail("unknown reference", r->name); return; }
    A.ptr = r->addr;
    switch (r->type) {
    case ',': case '_': emit((x - r->addr - 2) & 0xff); break;
    case '.': case '-': emit(x & 0xff); break;
    case ';': case '=': emit2(x); break;
    default: emit2((x - r->addr - 2) & 0xffff); break; /* '!' '?' and bare JSI */
    }
  }
}

int uxnasm_assemble(const char *src, int len, Uint8 *rom, int *rom_len) {
  A.rom = rom;
  A.ptr = A.top = 0x100;
  A.nlabels = A.nrefs = A.nmacros = A.nlambdas = 0;
  A.line = 1;
  A.failed = 0;
  s_cpy(A.scope, "on-reset");
  uxnasm_error[0] = 0;
  uxnasm_error_line = 0;
  for (int i = 0; i < 0x10000; i++) rom[i] = 0;
  assemble_range(src, src + len);
  if (!A.failed && A.nlambdas) fail("unterminated lambda", "");
  if (!A.failed) resolve();
  *rom_len = A.top - 0x100;
  return !A.failed;
}
