# Variant (`case`) objects with scalar branch fields: construction per branch, discriminant test,
# branch field read. `objects_variant_string` is the same shape with a string branch; it was a known
# gap once, and has been in the corpus proper since it was fixed.
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
