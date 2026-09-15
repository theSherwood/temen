# The nim differential corpus

Each `.nim` file here is compiled and run **twice** — once on Temen, once by native `nimony c --run` —
and the two outputs are diffed byte for byte by `nim_differential_corpus` in `../nim_e2e.rs`.

**Adding a case is adding a file.** There is no expected value to write down: the native toolchain is
the oracle, so a case cannot bake in a wrong constant, and a case that stops compiling natively is
reported as a corpus bug rather than a Temen one.

Rules for a case:

- **Deterministic.** No clock, no addresses, no PRNG without a fixed seed, no iteration order nim does
  not itself pin. The whole value of the suite is that a diff means a real defect.
- **Inside nimony's subset.** It is a strict subset of Nim: no `echo`, `$seq`, `toHex`; `[]` on a seq,
  `hasKey` and `parseInt` are `.raises` and need a `try`/`except`. If the driver reports "does not run
  under native nimony", the program is wrong, not Temen.
- **Print something.** A case that prints nothing compares nothing.
- **Narrow.** One construct family per file, so a diff points at a cause.

The class this targets is *compiles, verifies, runs — wrong bytes*. Both `$3.14` printing
`17.966570549813729` (#1472) and half of libm being unbound (#1375) lived in `main` behind suites that
happened not to look.

## `known_gaps/`

Programs that are **expected** to diverge, each naming the issue in its header comment. They are
still compiled and run every time: a gap that quietly starts working is reported as a failure, with
instructions to promote the file into the main corpus and close its issue. The directory *is* the
expectation — there is no per-case enum to keep in sync.

Put a case here only when the divergence is understood and tracked. An unexplained diff belongs in
the main corpus, red, until someone explains it.
