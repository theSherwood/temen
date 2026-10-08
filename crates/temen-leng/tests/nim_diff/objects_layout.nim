# Object layout as C lays it out (#2201): every object's size and every field's offset, which native
# nimony takes from its C compiler. An object that crosses into C or Rust code, or onto a wire with a
# C layout, reads the wrong fields unless both agree.
import std/syncio

type
  Kind = enum kA, kB
  Small = distinct uint16

  Padded = object
    a: uint8
    b: int32
    c: int64
  Nested = object
    x: uint8
    p: Padded
    y: uint16
  WithArray = object
    a: uint8
    arr: array[3, int32]
    b: uint8
  Floats = object
    a: uint8
    f: float32
    d: float64
  Scalars = object
    b: bool
    c: char
    k: Kind
    s: Small
    i: int32
  Union {.union.} = object
    a: uint8
    b: int64
    c: array[3, uint16]
  Variant = object
    tag: uint8
    case kind: Kind
    of kA: x: uint8
    of kB: y: int64
  Packed {.packed.} = object
    a: uint8
    b: int32
    c: int64
  Aligned = object
    a: uint8
    b {.align: 16.}: int32
  Base = object of RootObj
    a: uint8
  Derived = object of Base
    b: uint8
  Tail = object
    n: int32
    data: UncheckedArray[int64]

proc off(base, field: pointer): string = $(cast[int](field) - cast[int](base))

proc go(): string =
  var p = default(Padded)
  result = "Padded " & $sizeof(Padded) & ": " & off(addr p, addr p.a) & " " & off(addr p, addr p.b) &
    " " & off(addr p, addr p.c) & "\n"
  var n = default(Nested)
  result.add "Nested " & $sizeof(Nested) & ": " & off(addr n, addr n.x) & " " & off(addr n, addr n.p) &
    " " & off(addr n, addr n.y) & "\n"
  var w = default(WithArray)
  result.add "WithArray " & $sizeof(WithArray) & ": " & off(addr w, addr w.arr) & " " &
    off(addr w, addr w.b) & " | array[2, Padded] " & $sizeof(array[2, Padded]) & "\n"
  var f = default(Floats)
  result.add "Floats " & $sizeof(Floats) & ": " & off(addr f, addr f.f) & " " & off(addr f, addr f.d) & "\n"
  var s = default(Scalars)
  result.add "Scalars " & $sizeof(Scalars) & ": " & off(addr s, addr s.c) & " " & off(addr s, addr s.k) &
    " " & off(addr s, addr s.s) & " " & off(addr s, addr s.i) & "\n"
  var u = default(Union)
  result.add "Union " & $sizeof(Union) & ": " & off(addr u, addr u.b) & " " & off(addr u, addr u.c) & "\n"
  var v = Variant(tag: 1, kind: kB, y: 7)
  result.add "Variant " & $sizeof(Variant) & ": " & off(addr v, addr v.kind) & " " & off(addr v, addr v.y) &
    " " & $v.y & "\n"
  var k = default(Packed)
  result.add "Packed " & $sizeof(Packed) & ": " & off(addr k, addr k.b) & " " & off(addr k, addr k.c) & "\n"
  var a = default(Aligned)
  result.add "Aligned " & $sizeof(Aligned) & ": " & off(addr a, addr a.b) & "\n"
  var d = Derived(a: 1, b: 2)
  result.add "Derived " & $sizeof(Base) & " " & $sizeof(Derived) & ": " & off(addr d, addr d.a) & " " &
    off(addr d, addr d.b) & " " & $(int(d.a) + int(d.b)) & "\n"
  var buf = default(array[4, int64])
  let t = cast[ptr Tail](addr buf)
  result.add "Tail " & $sizeof(Tail) & ": " & off(t, addr t.data) & "\n"

write(stdout, go())
