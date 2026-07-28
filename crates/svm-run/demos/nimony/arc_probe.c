// NIM.md Phase 1, step 1 — the codegen-shape probe.
//
// This is NOT hand-written application C: every construct here mirrors a specific pattern
// that nimony/hexer emits into Leng and that lengc turns into C (doc/leng-spec.md). The point
// is to retire the load-bearing Phase-1 risk *before* a full nimony bootstrap is available:
// does Nim-SHAPED codegen survive the LLVM on-ramp (LLVM.md) and run bit-identically on the
// SVM interpreter and JIT, under window+mask confinement (INVARIANTS.md §2)?
//
// The four modeled patterns (NIM.md §1 "Runtime shape"):
//   1. ARC/ORC  — a heap object with an explicit refcount; `inc`/`dec` and a `=destroy` hook
//                 called as an ordinary function when the count hits 0 (hexer injects these;
//                 there is no tracing GC — the destructor is just a call in Leng).
//   2. Exceptions — the error-flag + goto model (`errv` bool + `onerr`): a raising call sets a
//                 flag, the caller branches to its cleanup label. No setjmp/longjmp.
//   3. Objects  — a tagged `object` with single inheritance: the base struct is the FIRST
//                 member (Leng models inheritance as first-child-is-parent), dispatched on a
//                 type tag — the shape method calls lower to.
//   4. seq/string — a heap `{len, cap, data}` payload grown with realloc, the Nim seq/string
//                 representation.
//
// Only names the on-ramp recognizes are used (printf/malloc/realloc/free/memcpy/memset —
// LLVM.md slices N/O/W/X); floats are deliberately omitted (the probe tests runtime *shape*,
// not float formatting — that is a separate ledger row in NIM.md §1). Deterministic output so
// the guest run diffs byte-for-byte against a native build.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// (1) ARC: refcounted heap object + a =destroy hook invoked as a plain call.
// ---------------------------------------------------------------------------
static long g_destroys = 0;   // observable side effect: how many =destroy hooks ran, and when.

typedef struct Node {
    long refcount;            // ARC book-keeping (hexer-injected).
    long value;
} Node;

static Node *node_new(long v) {
    Node *n = (Node *)malloc(sizeof(Node));
    n->refcount = 1;          // ARC: born with one owner.
    n->value = v;
    return n;
}
static void node_destroy(Node *n) {   // the `=destroy` hook.
    g_destroys += 1;
    free(n);
}
static void node_incref(Node *n) { n->refcount += 1; }
static void node_decref(Node *n) {    // ARC `dec`: destroy at zero.
    n->refcount -= 1;
    if (n->refcount == 0) node_destroy(n);
}

// ---------------------------------------------------------------------------
// (2) Exceptions: the error-flag + goto lowering. `checked_div` "raises" on /0 by
//     setting *err and returning; the caller tests the flag and jumps to cleanup.
// ---------------------------------------------------------------------------
static long checked_div(long a, long b, int *err) {
    if (b == 0) { *err = 1; return 0; }   // `raise` → set errv.
    return a / b;
}

// Sums a/b[i] over an array, ARC-owning a scratch Node across the raising call so the
// error path must run the destructor (cleanup-on-unwind, the reason ARC+exceptions interact).
static long sum_divs(long a, const long *bs, long n, int *err) {
    long acc = 0;
    Node *scratch = node_new(0);          // owned local; must be destroyed on every exit edge.
    for (long i = 0; i < n; i++) {
        int e = 0;
        long q = checked_div(a, bs[i], &e);
        if (e) {                          // `onerr`: propagate + run cleanup.
            *err = 1;
            goto cleanup;
        }
        scratch->value += q;
        acc += q;
    }
cleanup:
    node_decref(scratch);                 // ARC dec on the unwind/normal path alike.
    return acc;
}

// ---------------------------------------------------------------------------
// (3) Objects: single-inheritance via first-member base + a type tag; a "method" that
//     dispatches on the tag — what Leng `dot`(…, inheritance-depth) + case lowers to.
// ---------------------------------------------------------------------------
typedef struct Shape { int tag; } Shape;                 // base.
typedef struct Circle { Shape base; long r; } Circle;    // base is the FIRST member.
typedef struct Square { Shape base; long side; } Square;

enum { TAG_CIRCLE = 1, TAG_SQUARE = 2 };

static long shape_area_x4(const Shape *s) {   // area*4 (integer; π≈ not needed for shape test).
    switch (s->tag) {
        case TAG_CIRCLE: { const Circle *c = (const Circle *)s; return 3 * c->r * c->r * 4; }
        case TAG_SQUARE: { const Square *q = (const Square *)s; return q->side * q->side * 4; }
        default: return -1;
    }
}

// ---------------------------------------------------------------------------
// (4) seq: a {len,cap,data} payload grown with realloc, then summed.
// ---------------------------------------------------------------------------
typedef struct Seq { long len, cap; long *data; } Seq;

static void seq_push(Seq *s, long v) {
    if (s->len == s->cap) {
        long ncap = s->cap ? s->cap * 2 : 4;
        s->data = (long *)realloc(s->data, (size_t)ncap * sizeof(long));
        s->cap = ncap;
    }
    s->data[s->len++] = v;
}
static long seq_sum(const Seq *s) {
    long acc = 0;
    for (long i = 0; i < s->len; i++) acc += s->data[i];
    return acc;
}

int main(void) {
    // (1)+(2): normal path and raising path; observe destructor counts.
    int err = 0;
    long ok_divs[] = {2, 3, 5};
    long r1 = sum_divs(30, ok_divs, 3, &err);          // 15 + 10 + 6 = 31, err stays 0.
    printf("sum_divs ok:   r=%ld err=%d\n", r1, err);

    err = 0;
    long bad_divs[] = {2, 0, 5};
    long r2 = sum_divs(30, bad_divs, 3, &err);         // 30/2=15 then raises at b=0; r=15, err=1.
    printf("sum_divs raise: r=%ld err=%d\n", r2, err);

    // ARC ref juggling: shared ownership, then release.
    Node *a = node_new(7);
    node_incref(a);            // rc=2
    node_decref(a);            // rc=1, no destroy
    node_decref(a);            // rc=0, destroy
    printf("destroys so far: %ld\n", g_destroys);   // 2 scratch + 1 = 3.

    // (3): object dispatch.
    Circle c = {{TAG_CIRCLE}, 5};
    Square q = {{TAG_SQUARE}, 4};
    printf("area*4 circle=%ld square=%ld\n",
           shape_area_x4((Shape *)&c), shape_area_x4((Shape *)&q));

    // (4): seq grow + sum (forces realloc across cap boundaries).
    Seq s = {0, 0, NULL};
    long total = 0;
    for (long i = 1; i <= 10; i++) { seq_push(&s, i * i); total += i * i; }
    long got = seq_sum(&s);
    printf("seq len=%ld sum=%ld (want %ld) ok=%d\n", s.len, got, total, got == total);
    free(s.data);

    // A single integer exit code that folds the invariants (byte-diff already covers stdout,
    // this makes the run's Outcome meaningful too): 0 iff everything held.
    int all_ok = (r1 == 31) && (err == 1 && r2 == 15) && (g_destroys == 3)
              && (shape_area_x4((Shape *)&c) == 300) && (shape_area_x4((Shape *)&q) == 64)
              && (got == total);
    return all_ok ? 0 : 1;
}
