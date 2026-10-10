# Nim conformance matrix

Which Nim features run on Temen (#956). Each row is a feature, the corpus cases that exercise it, and
its status:

- **✅**: the cases are in this directory. `nim_differential_corpus` (`../nim_e2e.rs`) builds each one
  with native nimony, runs it on Temen, and fails if the two outputs differ by a byte.
- **❌ #N**: the case is in [`known_gaps/`](known_gaps/), its header names issue #N, and it still
  diverges. The corpus fails when it starts to match, so the row and the issue get updated.
- **n/a**: nimony itself rejects the construct, so there is no native output to compare. The note
  gives nimony's error.

`the_matrix_matches_the_corpus` (`../nim_e2e.rs`) checks this file against the two directories
without the toolchain. Every case a row names is where its status says, a gap's header names the
row's issue, an n/a row names no case, and every corpus case is in exactly one row. So adding a case
means adding it to a row.

## Values and types

| Feature | Cases | Status |
|---|---|---|
| Integer arithmetic, `div`/`mod` signs, shifts | `int_arith` | ✅ |
| Integer widths, signedness, unsigned comparison | `int_widths`, `int_compare`, `uint64_bounds` | ✅ |
| Sub-word integers (`int8`…`uint16`): wrapping, calls, conversions, consumers | `subword_signed`, `subword_unsigned`, `subword_signed_to_unsigned`, `subword_calls`, `subword_canonical_consumers` | ✅ |
| `char`, and UTF-8 bytes as unsigned | `chars`, `utf8_bytes` | ✅ |
| Floats: `$`, formatting, `std/math` | `float_dollar`, `float_format`, `float_math` | ✅ |
| Enums: names, ordinals, explicit values, string values | `enums` | ✅ |
| `distinct` types, module-level `const` | `distinct_const` | ✅ |
| `set[enum]`, sparse ordinals included | `sets_enum`, `sets_enum_sparse` | ✅ |
| Tuples | `tuples` | ✅ |
| Strings, `strutils` | `strings` | ✅ |
| Slices and backward indexes | `slices` | ✅ |
| Arrays | `arrays` | ✅ |
| `seq` | `seqs`, `seq_of_strings` | ✅ |
| Nested containers | `nested_containers` | ✅ |
| Raw pointers: `addr`, `cast`, `ptr UncheckedArray` | `pointers` | ✅ |
| `$` on a `char` or a `seq` | — | n/a: nimony reports `Type mismatch` |
| Iterating an enum type, `for c in Color` | — | n/a: nimony reports `Type mismatch`; `low(Color)..high(Color)` works |
| Wrapping operators such as `+%` | — | n/a: nimony reports `undeclared identifier: '+%'` |

## Objects

| Feature | Cases | Status |
|---|---|---|
| Value objects, nested objects | `objects_plain`, `objects_nested` | ✅ |
| Object layout as C lays it out | `objects_layout` | ✅ |
| A module-level aggregate initializer | `global_aggregate_init` | ✅ |
| `ref object` | `objects_ref`, `objects_ref_string` | ✅ |
| An object with a `seq` field | `objects_seq_field` | ✅ |
| Variant (`case`) objects | `objects_variant`, `objects_variant_string` | ✅ |
| Lifetime hooks: `=destroy`, `=copy` | `hooks` | ✅ |
| Methods: dynamic dispatch with overrides | `methods_dispatch` | ✅ |
| Inheritance: a derived type with no method override of its own | `objects_inheritance` | ❌ #2260 |
| Copying a derived value object whose base has a `string` field | — | n/a: nimony reports `expected expression but got: (baseobj …)` |

## Control flow and procs

| Feature | Cases | Status |
|---|---|---|
| Branches, loops, early exit, recursion | `control_flow` | ✅ |
| `case` over strings, labeled `block` | `control_case_blocks` | ✅ |
| Exceptions (nimony's `ErrorCode` model) | `exceptions` | ✅ |
| `defer`, `try`/`finally` | `defer_finally` | ✅ |
| Parameters: `var`, defaults, named arguments | `params` | ✅ |
| `openArray` and `varargs` parameters | `openarray_varargs` | ✅ |
| User-defined operators, overloading | `operators` | ✅ |
| Procs as values | `proc_types` | ✅ |
| Closures | `closures` | ✅ |
| Inline iterators | `iterators` | ✅ |
| Closure iterators | `iterators_closure` | ✅ |
| Threads (`std/rawthreads`) | `threads` | ✅ |

## Generics and metaprogramming

| Feature | Cases | Status |
|---|---|---|
| Generic procs and types | `generics` | ✅ |
| Concepts, constrained generics | `concept_constraints`, `generic_signed_compare` | ✅ |
| Templates | `templates` | ✅ |
| Macros | `macros` | ✅ |
| Compile-time evaluation, `when` | `compile_time` | ✅ |
| Converters between `distinct` types | `converters` | ✅ |
| `+` on a generic `T: SomeNumber` | — | n/a: nimony reports `Type mismatch`; comparisons on `T: SomeInteger` work (`generic_signed_compare`) |
| A converter to a builtin type | — | n/a: nimony reports `cannot attach converter to type int64` |
| `newProc`, `newLit` in a macro | — | n/a: nimony's `std/macros` has neither; `newTree` and `ident` work (`macros`) |

## Standard library

| Feature | Cases | Status |
|---|---|---|
| `std/algorithm` | `std_algorithm` | ✅ |
| `std/atomics` | `std_atomics` | ✅ |
| `std/base64`, `std/md5` | `std_base64_md5` | ✅ |
| `std/bitops` | `std_bitops` | ✅ |
| `std/complex` | `std_complex` | ✅ |
| `std/deques`, `std/heapqueue` | `std_deques_heapqueue` | ✅ |
| `std/hashes`, `std/sets` | `std_hashes_sets` | ✅ |
| `std/json` | `std_json` | ✅ |
| `std/lexbase` | `std_lexbase` | ✅ |
| `std/options`, `std/intsets` | `std_options_intsets` | ✅ |
| `std/parseopt` | `std_parseopt` | ✅ |
| `std/parseutils` | `std_parseutils` | ✅ |
| `std/paths`, `std/pathnorm` | `std_paths_pathnorm` | ✅ |
| `std/random` from a fixed seed | `std_random_seeded` | ✅ |
| `std/sequtils` | `std_sequtils` | ✅ |
| `std/setutils` | `std_setutils` | ✅ |
| `std/sha1` | `std_sha1` | ✅ |
| `std/streams` | `std_streams` | ✅ |
| `std/strtabs` | `std_strtabs` | ✅ |
| `std/tables`: `Table`, with `HashSet` | `tables_sets` | ✅ |
| `std/tables`: `OrderedTable` | `std_tables_ordered` | ✅ |
| `std/times` over fixed instants | `std_times_pure` | ✅ |
| `std/unicode` | `std_unicode` | ✅ |
| `std/varints` | `std_varints` | ✅ |
| `std/widestrs` | `std_widestrs` | ✅ |
| `std/wordwrap`, `std/editdistance` | `std_wordwrap_editdistance` | ✅ |
| `std/tables`: `CountTable` | — | n/a: nimony reports `undeclared identifier: initCountTable` |
| `std/strformat` | — | n/a: nimony ships no `std/strformat` |
