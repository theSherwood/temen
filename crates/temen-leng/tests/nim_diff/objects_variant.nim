# Variant (`case`) objects with scalar branch fields: construction per branch, discriminant test,
# branch field read. The string-branch shape is a known gap — see known_gaps/objects_variant_string.
import std/syncio

type
  Kind = enum kInt, kBig
  Node = object
    case kind: Kind
    of kInt: ival: int
    of kBig: bval: int64

proc describe(n: Node): string =
  case n.kind
  of kInt: "int:" & $n.ival
  of kBig: "big:" & $n.bval

proc go(): string =
  let a = Node(kind: kInt, ival: 42)
  let b = Node(kind: kBig, bval: 9000000000'i64)
  describe(a) & "|" & describe(b) & "|" & $(a.kind == kInt)

write(stdout, go() & "\n")
