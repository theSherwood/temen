# TYPESCRIPT.md — a static-first TypeScript frontend for SVM

Status: **design doc, pre-implementation**, written 2026-08-04. This records the
load-bearing language-subset and object-model decisions for compiling TypeScript →
SVM IR ahead of any code. It leans on the frontend trust model (`DESIGN.md` §2a,
`FRONTEND.md` §1), the guest-GC contract (`GC.md`), and the guest-JIT posture
(DESIGN.md §3). Prior art in-tree: the C frontend (`FRONTEND.md`), nimony
(`NIM.md`), JACL (external, vendors svm; guest-owned object model + GC), and
QuickJS-via-LLVM as the *full-engine* comparison point
(`crates/svm-run/demos/quickjs/`).

Fold settled sections into `DESIGN.md` and drop this file once implementation
lands — the repo convention (cf. the former `WASM.md`/`SCHEDULING.md`).

## 0. TL;DR

- **Goal.** Run real TypeScript *scripts* from the ecosystem fast on SVM — not the
  whole npm ecosystem, not full JS compatibility, and not an AssemblyScript-style
  new-language-in-TS-syntax. `number` stays f64. Well-typed code gets an unboxed
  fast path; the dynamic tail (`any`, unions, dictionaries) gets one boxed slow
  path. AOT, type-directed: **no runtime respecialization** — no hidden classes,
  no inline caches, no shape-transition trees, no adaptive machinery.
- **The one big cut is the sealed object model** (§2): object shape is fixed at
  allocation — mutate field *values* freely, never the field *set*. This deletes
  the expensive part of a JS engine (shapes/ICs/megamorphic dispatch), which is
  also the least trustworthy part. Everything else (`with`, `arguments`, `eval`,
  `Proxy`) is a cheap corollary cut.
- **Two representation worlds, chosen statically by type** (§3): closed
  known-key types → flat structs with fixed-offset access; index-signature /
  `Map` / `any` types → hash tables + boxed tagged values. The worlds meet only
  at explicit, checked crossings. Representation is *assigned* by the static
  type, never *discovered* per access site.
- **Layout is decision "1b"** (§4, the settled choice): canonical per-type layout
  (fields sorted lexicographically by name — a pure function of the structural
  type, no shared state), direct offset loads wherever the access site's static
  type is concrete, and a per-(concrete-type, interface) offset table — one extra
  load, fully static — at genuinely polymorphic interface-typed sites.
  Alternatives (global slot coloring, whole-program monomorphization,
  interface-as-hash-table) were considered and rejected; §4a records why, so
  they aren't relitigated by accident.
- **Trust posture:** the frontend and the TS runtime (GC, hash tables, boxing)
  are guest code, **outside the escape-TCB** (§5). The verifier re-checks emitted
  IR; the masking lowering confines every access. A frontend or runtime bug
  corrupts the guest's own world, never the host.

## 1. Goal & posture

**What "TypeScript support" means here.** Run unmodified well-typed TS source
from the ecosystem — compute-shaped scripts, plugins, kernels — with performance
that beats an interpreted engine by a wide margin on typed code. Programs that
lean on the banned dynamism (§2) are rejected at compile time with a clear
error; that is the compatibility price, accepted deliberately. This sits between
the two poles already explored: QuickJS-on-svm (full semantics, no speed) and
AssemblyScript (speed, but a different language — loses `number`, loses the
ecosystem). We keep TS's types *and* its numeric semantics, and spend the
compatibility budget only on the dynamism that actually costs machinery.

**`number` stays f64.** No `i32`/`i64` dialect types at the language level —
that forks the language. Integer-level speed comes from analysis: a `number`
proven to be a bounded integer (array index, loop counter) may be unboxed to i32
internally, guarded where the proof is conditional. Same source, same semantics.

**AOT only; open to a guest JIT.** Types are the static replacement for runtime
type feedback — analyze once, emit specialized IR, done. Guards are *emitted*,
never adaptively installed. Nothing in this design may require whole-program
closure in a way that forecloses a later guest JIT compiling new code at
runtime (svm supports guest JIT); this constraint is what settled §4.

**`any` is kept.** Ecosystem TS is shot through with `any`, `as`, and untyped
values; banning them would shrink runnable code drastically for little gain.
What makes `any` expensive in a real engine is not the unknown type — it's that
`any` can reach dynamic shapes, traps, and `eval`. With those banned (§2), `any`
degrades to "boxed tagged value → generic slow path": a tag dispatch and maybe a
hash lookup. Cheap dynamism, kept; fatal dynamism, cut.

## 2. The subset: sealed object model + corollary cuts

**Sealed shapes (the load-bearing cut).** An object's shape — its set of
properties and their layout — is fixed at allocation and never changes:

- ✅ mutating existing property values (`p.x = 5`, `p.x++`)
- ❌ adding a property after construction (`p.z = 9`)
- ❌ `delete p.x`
- ❌ prototype mutation (`__proto__ =`, `Object.setPrototypeOf`, reassigning the chain)
- ❌ accessor properties (getters/setters) — at most a tightly limited static form later
- ❌ property attributes (`writable`/`enumerable`/`configurable` machinery)

Objects are structs; property access is a fixed-offset load. This is what
deletes hidden classes, shape-transition trees, inline caches, and megamorphic
handlers — most of an engine's hot-path complexity and subtle-bug surface.
Well-typed TS already lives inside this box: adding an undeclared property is a
type error today. The one genuine loss — objects-as-dictionaries
(`counts[word] = …`) — is not lost but *retyped*: dynamic keys must say so in
the type (`Map`, `Record<string, T>`, index signatures) and get the hash-table
representation (§3).

**Corollary cuts** (each cheap once shapes are sealed; all mandatory):

- `eval` / `new Function` — gates AOT at all; a resident compiler is exactly the
  machinery we refuse to carry.
- `with` — makes name binding non-static.
- `arguments` — forces an arguments-object and aliasing rules into the calling
  convention; rest params cover it.
- `Proxy` / `Reflect` traps — puts an "is this exotic?" branch on every access.
- getters/setters as above.

**Kept:** closures, classes, prototype-based *method dispatch on a frozen
chain*, exceptions, generators/async (lower to svm fibers/continuations —
detail deferred), strings, arrays, `Map`/`Set`, JSON, structural typing,
generics, unions, `any`.

## 3. Two worlds, one explicit seam

Representation follows the **static type** — decided at compile time, never
discovered per access:

- **Typed world** — closed known-key types (classes, object literals with fixed
  fields, interfaces/type aliases over them): flat struct layout, offset loads,
  statically-dispatched methods. No hash tables, no representation branches.
- **Dynamic world** — `any`, unions of unrelated types, index signatures,
  `Record<string, V>`, `Map`: boxed tagged values; string-keyed hash tables for
  dictionary objects. `Record` and `Map` bottom out on the same table.

The rule is **keyword-blind**: `interface` vs `type` carries no representational
meaning (the ecosystem chose between them stylistically; `.d.ts` files use
`interface` for plain closed data everywhere). What matters is what the type
*says*: closed field set → struct; dynamic keys → table.

**Crossings are explicit coercions, at known sites:**

- **typed → dynamic**: box the struct reference with a tag. Cheap, always
  sound; the dynamic world reads it through generic ops that consult its shape
  descriptor (§4).
- **dynamic → typed**: the guarded direction. Either unbox (tag says it already
  is a struct of a compatible shape — a check and a pointer) or
  materialize/validate into a struct (a copy — which breaks reference identity
  with the source, so it is only legal at sites the user wrote as
  parse/validate/checked-cast, the way TS codebases already treat `JSON.parse`
  boundaries). Never implicit.

The banned outcome, restated as the design's one hard rule: **no access site
ever branches on representation.** A site typed against a struct type sees only
structs; a site typed `any` sees only boxed values. The moment a single site can
see both, we are rebuilding inline caches — the machinery this design exists to
delete. (V8's "dictionary mode" demotion is the cautionary tale: two
representations are fine; *silent per-site mixing* is what forces IC machinery.
Engines mix because JS has no static types. We have types; we get to assign.)

## 4. Layout — the settled decision (owner, 2026-08-04: "1b")

The problem: TS is structurally typed, so structurally-compatible types with
*different field sets* flow into the same interface-typed positions
(`{x,y,z}` used as `{x,y}`). Struct layouts can't all agree on offsets for
shared fields under any pure *ordering* rule (adding a field shifts the rest),
so interface-typed access needs a plan. The settled plan:

**(1) Canonical per-type layout, stateless and deterministic.** A concrete
type's layout is a pure function of its structural shape: fields sorted
**lexicographically by property name** (UTF-8 byte order), each field one
8-byte slot (MVP; packing is a later, measured optimization). No registry, no
load-order dependence, no shared state: any compiler — the AOT frontend today, a
guest JIT later — computes the same layout from the type alone. This is also
Invariant-10-shaped: layout identity is the structural shape, never a name or a
registry entry.

**(2) Concrete sites: direct offset loads.** Wherever the access site's static
type is a concrete class/struct type — the overwhelmingly common case in typed
code — the compiler knows the exact layout and emits a fixed-offset load.
No convention across types is even consulted.

**(3) Polymorphic interface sites: per-(type, interface) offset tables.** Where
the static type is an interface and multiple concrete layouts genuinely reach
it, access goes through an offset table: the interface's own fields, in its own
canonical order, index a small array giving the byte offset of each field in
the concrete type; the concrete type is found from the object's shape-descriptor
header word. Cost: one extra dependent load per access. Fully static — tables
are computed per (concrete type, interface) pair at compile time (or appended
by a guest JIT for new types), never installed adaptively, never invalidated.
No copies anywhere in §4: reference identity, mutation-sharing, and `===`
behave normally across interface views.

**Shape descriptor.** Every heap object carries one header word referencing its
shape descriptor: `{ kind: struct | dict, sorted field list, field count }`, the
anchor for offset tables, dynamic-world generic access, and GC tracing. Interned
structurally (same field set → same descriptor), which makes the §3
dynamic→typed unbox check a descriptor compare. Interning is the one piece of
shared runtime state; it is an append-only intern table (string/shape → id) —
monotonic and dumb, safe for a guest JIT to extend.

### 4a. Rejected alternatives (recorded so they stay rejected)

- **Global slot coloring ("1a").** Assign every field name a global slot;
  objects allocate slots-with-holes; all offsets agree universally, every access
  everywhere a direct load. Rejected: space holes, a graph-coloring pass in the
  hot design, and global slot state that fights runtime code loading — a clever
  pass where a boring table suffices (Invariant 1).
- **Whole-program monomorphization ("#3").** Compile each interface-generic
  function once per concrete layout. Fastest code, but buys it by assuming a
  closed world — every concrete type reaching every site known at compile time —
  which forecloses guest-JIT-loaded code. Rejected for the openness constraint;
  remains available later as a *local* optimization where the compiler proves
  the reaching set (an optimization, not the semantics).
- **`interface` = hash table, `type` = struct.** Attaches representation to a
  keyword the ecosystem chose stylistically (all of `.d.ts` reality would land
  on the slow path), and structural assignability pushes structs into
  interface-typed positions constantly — putting the representation seam on
  every ordinary call boundary, i.e. per-site mixing, i.e. inline caches again.
- **Fully-static subset, no `any`.** Once shapes are sealed, banning `any` buys
  only the removal of the boxed slow path — incremental — while the
  compatibility cost is large (ecosystem TS is full of `any`). Rejected as the
  wrong knob turned hard.

## 5. Trust posture & invariants respected

- **Outside the escape-TCB** (`DESIGN.md` §2a, `FRONTEND.md` §1): the TS
  compiler is a frontend like chibicc/nimony — the verifier re-checks everything
  it emits; the masking lowering (Invariant 2) confines every access. The TS
  *runtime* (GC, hash tables, boxing helpers, shape interning) is guest code in
  the guest window, per the `GC.md` division of labor: svm provides root
  enumeration only, the guest owns its heap. `GC.md`'s high-byte payload-mask
  support (`(tag << 56) | offset` roots) is the intended hook for the boxed
  value representation — the tagging scheme should be chosen to fit it (§6).
- **Invariant 1 (small core):** every choice above prefers a table over a pass
  and a static rule over adaptive machinery; nothing here adds host-side
  surface at all.
- **Invariant 9 (interpreter is the oracle):** the frontend targets the same IR
  as every other frontend; differential testing of backends is inherited, and
  the TS compiler itself gets an oracle by differential-testing compiled output
  against an existing JS engine on the shared subset (QuickJS is already
  in-tree as a guest).
- **Invariant 10 (identity is structural):** shape descriptors are interned
  structural shapes; layout is a pure function of shape; no nominal registries.

## 6. Open questions (decide before/while building)

1. **Boxed value representation.** NaN-boxing vs. high-byte-tagged window
  offsets. The `GC.md` payload mask constrains tags to the top byte — leaning
  tagged-offset; needs a concrete tag map (f64, i32, struct-ref, dict-ref,
  string, null/undefined, bool).
2. **Struct field slots.** Uniform 8-byte slots is the MVP call above; when to
  pack (f32 arrays, bools), and whether `number`-proven-i32 fields narrow.
3. **Strings.** Immutable, but representation (flat UTF-8 vs. rope vs. interned)
  and the `string` ↔ dynamic-world story.
4. **Generics.** Erased-with-boxing vs. reachable-set specialization (the local,
  non-closed-world kind) — likely erased first, specialize by measurement.
5. **Async/generators → fibers.** Mapping to svm continuations (§12/D22);
  which of `Promise` semantics survive the subset.
6. **Checker reuse.** Whether to consume `tsc`'s checked AST/types (heavy
  dependency, exact ecosystem semantics) or a from-scratch checker for the
  subset (small, but a semantics fork risk). Needs a scoping spike.
7. **The compatibility probe.** Before deep investment: take representative
  target scripts, measure how much runtime sits in code the typed fast path
  covers vs. the boxed tail — the go/no-go number for the whole effort.

## 7. Non-goals

- Full JS/TS compatibility; npm-at-large; Node API surface.
- `eval`/`Function`, `with`, `arguments`, `Proxy`, dynamic shapes, accessor
  machinery (§2) — by design, not by omission.
- A resident optimizing runtime: no ICs, no tiering, no deopt. If a guest JIT
  arrives later it compiles *new code* under the same static rules; it does not
  respecialize old code.
- Beating a tuned adaptive JIT (V8) on dynamic code. The bar is beating
  interpreted engines decisively on typed code while running the dynamic tail
  correctly.
