/* The Uxn CPU — see uxn.h. One `switch` over the 5 opcode bits; the three mode bits are handled
 * uniformly: `2` widens every stack operand to a short, `r` swaps the roles of the two stacks, and `k`
 * pops from a scratch copy of the stack pointer so the operands stay in place. Addresses wrap at 16
 * bits and stacks wrap at 8 (circular, like the reference and uxn.wasm), so no instruction can fault. */
#include "uxn.h"

#define RAM_MASK 0xffff

/* Pop `n` (1|2) bytes from stack `s` through the pop pointer `sp`; push onto `s` through its real ptr. */
static inline Uint16 pop(Stack *s, Uint8 *sp, int n) {
  Uint16 v = s->dat[(Uint8)(*sp - 1)];
  *sp -= 1;
  if (n == 2) {
    v |= (Uint16)(s->dat[(Uint8)(*sp - 1)] << 8);
    *sp -= 1;
  }
  return v;
}
static inline void push(Stack *s, Uint16 v, int n) {
  if (n == 2) s->dat[s->ptr++] = (Uint8)(v >> 8);
  s->dat[s->ptr++] = (Uint8)v;
}
/* `mask` bounds the second byte of a short: the whole address space (0xffff) or, for the zero-page
 * ops, the page (0xff) — both wrap. */
static inline Uint16 ld(Uint8 *ram, Uint16 a, int n, Uint16 mask) {
  Uint16 v = ram[a];
  if (n == 2) v = (Uint16)(v << 8 | ram[(a + 1) & mask]);
  return v;
}
static inline void st(Uint8 *ram, Uint16 a, Uint16 v, int n, Uint16 mask) {
  if (n == 2) {
    ram[a] = (Uint8)(v >> 8);
    ram[(a + 1) & mask] = (Uint8)v;
  } else
    ram[a] = (Uint8)v;
}

int uxn_eval(Uxn *u, Uint16 pc) {
  Uint8 *ram = u->ram;
  if (!pc || u->dev[0x0f]) return 0;
  for (;;) {
    Uint8 ins = ram[pc++];
    int n = ins & 0x20 ? 2 : 1; /* operand width in bytes */
    Stack *s = ins & 0x40 ? &u->rst : &u->wst;
    Stack *o = ins & 0x40 ? &u->wst : &u->rst; /* the "other" stack (JSR/STH) */
    Uint8 kp = s->ptr, *sp = ins & 0x80 ? &kp : &s->ptr;
    Uint16 a, b, c;
    switch (ins & 0x1f) {
    case 0x00:
      switch (ins & 0xe0) {
      case 0x00: /* BRK */ return 1;
      case 0x20: /* JCI */
        if (!pop(&u->wst, &u->wst.ptr, 1)) { pc += 2; break; }
        /* fall through */
      case 0x40: /* JMI */ pc += PEEK2(ram + pc) + 2; break;
      case 0x60: /* JSI */ push(&u->rst, pc + 2, 2); pc += PEEK2(ram + pc) + 2; break;
      default: /* LIT */ push(s, ld(ram, pc, n, RAM_MASK), n); pc += n; break;
      }
      if (!pc) return 0;
      break;
    case 0x01: /* INC */ a = pop(s, sp, n); push(s, a + 1, n); break;
    case 0x02: /* POP */ pop(s, sp, n); break;
    case 0x03: /* NIP */ b = pop(s, sp, n); pop(s, sp, n); push(s, b, n); break;
    case 0x04: /* SWP */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, b, n); push(s, a, n); break;
    case 0x05: /* ROT */
      c = pop(s, sp, n); b = pop(s, sp, n); a = pop(s, sp, n);
      push(s, b, n); push(s, c, n); push(s, a, n);
      break;
    case 0x06: /* DUP */ a = pop(s, sp, n); push(s, a, n); push(s, a, n); break;
    case 0x07: /* OVR */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, a, n); push(s, b, n); push(s, a, n); break;
    case 0x08: /* EQU */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, a == b, 1); break;
    case 0x09: /* NEQ */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, a != b, 1); break;
    case 0x0a: /* GTH */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, a > b, 1); break;
    case 0x0b: /* LTH */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, a < b, 1); break;
    case 0x0c: /* JMP */
      a = pop(s, sp, n);
      pc = n == 2 ? a : (Uint16)(pc + (Sint8)a);
      if (!pc) return 0;
      break;
    case 0x0d: /* JCN */
      a = pop(s, sp, n);
      b = pop(s, sp, 1);
      if (b) pc = n == 2 ? a : (Uint16)(pc + (Sint8)a);
      if (!pc) return 0;
      break;
    case 0x0e: /* JSR */
      a = pop(s, sp, n);
      push(o, pc, 2);
      pc = n == 2 ? a : (Uint16)(pc + (Sint8)a);
      if (!pc) return 0;
      break;
    case 0x0f: /* STH */ a = pop(s, sp, n); push(o, a, n); break;
    case 0x10: /* LDZ */ a = pop(s, sp, 1); push(s, ld(ram, a, n, 0xff), n); break;
    case 0x11: /* STZ */ a = pop(s, sp, 1); b = pop(s, sp, n); st(ram, a, b, n, 0xff); break;
    case 0x12: /* LDR */ a = pop(s, sp, 1); push(s, ld(ram, (Uint16)(pc + (Sint8)a), n, RAM_MASK), n); break;
    case 0x13: /* STR */ a = pop(s, sp, 1); b = pop(s, sp, n); st(ram, (Uint16)(pc + (Sint8)a), b, n, RAM_MASK); break;
    case 0x14: /* LDA */ a = pop(s, sp, 2); push(s, ld(ram, a, n, RAM_MASK), n); break;
    case 0x15: /* STA */ a = pop(s, sp, 2); b = pop(s, sp, n); st(ram, a, b, n, RAM_MASK); break;
    case 0x16: /* DEI */
      a = pop(s, sp, 1);
      if (n == 2) {
        b = uxn_dei(u, (Uint8)a);
        push(s, (Uint16)(b << 8 | uxn_dei(u, (Uint8)(a + 1))), 2);
      } else
        push(s, uxn_dei(u, (Uint8)a), 1);
      break;
    case 0x17: /* DEO */
      a = pop(s, sp, 1);
      b = pop(s, sp, n);
      if (n == 2) {
        u->dev[(Uint8)a] = (Uint8)(b >> 8);
        uxn_deo(u, (Uint8)a);
        u->dev[(Uint8)(a + 1)] = (Uint8)b;
        uxn_deo(u, (Uint8)(a + 1));
      } else {
        u->dev[(Uint8)a] = (Uint8)b;
        uxn_deo(u, (Uint8)a);
      }
      break;
    case 0x18: /* ADD */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, a + b, n); break;
    case 0x19: /* SUB */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, a - b, n); break;
    case 0x1a: /* MUL */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, (Uint16)((unsigned)a * b), n); break;
    case 0x1b: /* DIV */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, b ? a / b : 0, n); break;
    case 0x1c: /* AND */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, a & b, n); break;
    case 0x1d: /* ORA */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, a | b, n); break;
    case 0x1e: /* EOR */ b = pop(s, sp, n); a = pop(s, sp, n); push(s, a ^ b, n); break;
    case 0x1f: /* SFT */
      b = pop(s, sp, 1);
      a = pop(s, sp, n);
      push(s, (Uint16)((a >> (b & 0xf)) << (b >> 4)), n);
      break;
    }
  }
}
