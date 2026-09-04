# Forth on Temen — a sectorforth-class Forth whose kernel is Temen IR

`forth.temt` is a Forth written **directly in Temen text IR** (no C, no chibicc runtime, ~2500
lines) whose every colon definition is compiled to a verified Temen IR unit through the §22
guest-driven `Jit` capability. Design, rationale, and status: issue #1214.

```sh
cargo run -p temen-run --bin temen-run -- crates/temen-run/demos/forth/forth.temt --stdin program.fs
```

The playground card (`browser/web/play.js`, "Forth") runs the same kernel on the bytecode engine
from the committed asset `browser/web/assets/forth.temen`, rebuilt by
`ONLY=forth bash scripts/rebuild-assets.sh`.

## The language

- **One type.** Every cell is `i64`. Flags are `0`/`1`. Numbers are decimal (`-7` ok).
- **The data stack is SSA.** `: sq ( n -- n ) dup * ;` becomes an IR function `(i64) -> (i64)`.
  The stack-effect comment is **required** on a named word: it is the function's signature, and the
  body is checked against it at `;` (`stack effect mismatch`). `dup drop swap over rot nip tuck`
  permute the compile-time virtual stack and emit nothing.
- **Control flow:** `if … else … then`, `begin … until`, `begin … again`, `begin … while … repeat`.
  Both arms / every loop iteration must leave the same stack depth. `recurse` calls the word itself.
- **Top level.** Each line is compiled as an anonymous unit, `Jit.install`ed, called, and
  uninstalled. Values left on the stack persist in a REPL stack between lines.
- **Words:** `+ - * / mod and or xor lshift rshift = <> < u< <= > >= 0= negate invert 1+ 1- abs
  min max`, `@ ! c@ c! +! cells`, `variable create allot here ,`, `constant`, `emit type cr space
  spaces . u.`, `." text"`, `s" text"` (pushes `addr len`), `' word` (pushes the xt = its `call.dyn`
  slot), `\` and `( … )` comments.
- **Constants:** `<value> constant name` fixes a value at definition time; each use loads it back. It
  is a read-only cell, so the value can come from any computation (`2 3 + constant five`).
- **Runtime dispatch (typed `execute`):** `execute0 ( xt -- )`, `execute1 ( x xt -- y )`,
  `execute2 ( a b xt -- y )` call an xt (from `' word`, or stored in a variable) at run time — a
  `call.dyn` on the runtime funcref, so deferred words and dispatch tables work. Untyped `execute`
  (a runtime-arity call) stays out of the typed model.
- **Deferred words:** `defer ( a b -- c ) name` forward-declares a word; `' impl is name` binds (and
  rebinds) it at run time. The declared effect is the word's signature; each call inlines a
  load-from-cell + `execute`, so a deferred word composes inside colon definitions.
- **Early return:** `exit` terminates the current colon definition. Every path (the guarded `exit`
  and the fall-through) must satisfy the word's stack effect.
- **Errors** are reported per line: `line <n>: <message> near <token>`, and the kernel resynchronizes
  at the next line so later definitions still compile.
- **Fibers:** `task ( xt -- f )`, `resume ( f x -- status y )` (status `0` = suspended, `1` =
  returned), `yield ( x -- y )`. A task body is any word `( x -- y )`; it may be resumed from any
  later line.
- **Threads:** `spawn ( xt x -- t )` runs a `( x -- y )` word on a new vCPU, `join ( t -- y )`;
  `wait ( addr expected -- status )` / `notify ( addr n -- woken )` are the futex; atomics:
  `atomic@ ( addr -- x )`, `atomic! ( x addr -- )`, `atomic+! ( n addr -- old )`,
  `atomic-xchg ( n addr -- old )`, `cas ( expected new addr -- old )`.
- **Not yet:** counted loops (`do`/`loop`/`+loop`/`i`) — the loop-carried index needs a return-stack
  region threaded through every branch, tracked as a focused follow-up on #1237. Permanently out (they
  need a runtime stack, against the static-stack design): dynamic stack effects `?dup`/`pick` and
  untyped (runtime-arity) `execute`. Also absent: floats.

## How it works

The kernel is the outer interpreter (tokenizer, dictionary, number parser), a binary IR emitter
(LEB128 + the `temen-encode` wire layout, built in the window), the control-structure compiler, and
the primitive table — which is declared in a Forth **prelude** inside the module with one hardwired
defining word, `prim <name> <kind> <nin> <nout> <payload>`. Everything else (`cr`, `.`, `min`, …)
is defined in Forth in that prelude. Fiber and thread bodies always start at the module-0
trampoline (func 66) with the word's xt in the `sp` slot, so every engine resolves the entry through
the primary table. See the header of `forth.temt` for the memory map and function index table.

Tests: `crates/temen-run/tests/forth.rs` (interp == Cranelift JIT differential on every transcript,
plus the bytecode engine) and `browser/tests/forth_asset.rs` (the committed asset through the
browser on-ramp).
