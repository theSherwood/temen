# KNOWN GAP — leng: a variant (`case`) object with a **non-scalar branch field** fails to link with
# "`dot` on a non-object". An all-scalar variant is fine (see ../objects_variant.nim); adding a
# `string` branch is what breaks it. Found by this corpus.
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
