# An object nested inside another object, at module scope *and* as a local. The module-level one goes
# through `const_aggregate_bytes`'s nested-`oconstr` arm (#1444); the local one through the proc path.
import std/syncio

type
  Inner = object
    a: int
    b: string
  Outer = object
    tag: string
    inner: Inner
    n: int

let g = Outer(tag: "g", inner: Inner(a: 7, b: "seven"), n: 1)

proc go(): string =
  let l = Outer(tag: "l", inner: Inner(a: 9, b: "nine"), n: 2)
  g.tag & g.inner.b & $g.inner.a & "|" & l.tag & l.inner.b & $l.inner.a & $(g.n + l.n)

write(stdout, go() & "\n")
