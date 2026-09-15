# A variant (`case`) object with a **non-scalar branch field**. The `string` branch is what gives the
# object a destructor, so hexer emits `=copy`/`=destroy` hooks that take a `(ptr Node)` and then write
# `(dot dest kind 0)` on it with no `deref` — C's `dest->kind`. leng had no case for a pointer-typed
# local as a `dot` base, so it fell through to the cross-module-symbol path and typed it as a bare
# integer (#1480). An all-scalar variant (see objects_variant.nim) has no hooks and always worked.
import std/syncio

type
  Kind = enum kInt, kStr
  Node = object
    case kind: Kind
    of kInt: ival: int
    of kStr: sval: string

proc describe(n: Node): string =
  case n.kind
  of kInt: "int:" & $n.ival
  of kStr: "str:" & n.sval

proc go(): string =
  let a = Node(kind: kInt, ival: 42)
  let b = Node(kind: kStr, sval: "hi")
  describe(a) & "|" & describe(b) & "|" & $(a.kind == kInt)

write(stdout, go() & "\n")
