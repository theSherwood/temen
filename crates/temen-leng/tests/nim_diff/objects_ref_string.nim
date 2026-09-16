# A `ref object` carrying a string — a heap object under ARC, aliased through a second binding so the
# refcount path and the field mutation both show up in the output.
import std/syncio

type Node = ref object
  label: string
  n: int

proc go(): string =
  let a = Node(label: "a", n: 1)
  let b = Node(label: "b", n: 2)
  var c = a
  c.n = c.n + 10
  a.label & $a.n & "|" & b.label & $b.n

write(stdout, go() & "\n")
