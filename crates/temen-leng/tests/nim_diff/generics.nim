# Generic procs and generic object types: `Box[T]` over three element types, a two-parameter
# `Pair[A, B]` swapped into `Pair[B, A]`, and a constrained `maxOf[T: SomeInteger]` over signed and
# unsigned widths. Arithmetic on a generic `T` does not resolve in nimony (it goes through concepts;
# see concept_constraints), so the constrained proc compares rather than adds.
import std/syncio
type
  Box[T] = object
    val: T
  Pair[A, B] = object
    first: A
    second: B
proc boxed[T](x: T): Box[T] = Box[T](val: x)
proc get[T](b: Box[T]): T = b.val
proc swapped[A, B](p: Pair[A, B]): Pair[B, A] = Pair[B, A](first: p.second, second: p.first)
proc maxOf[T: SomeInteger](xs: openArray[T]): T =
  result = xs[0]
  for x in xs:
    if x > result: result = x
var s = $get(boxed(7)) & "|" & get(boxed("str")) & "|" & $get(boxed(2.5))
let p = swapped(Pair[int, string](first: 1, second: "one"))
s.add "|" & p.first & $p.second & "|" & $maxOf([3, 9, 4]) & "|" & $maxOf(@[7'u8, 2'u8]) & "|" & $maxOf([-5'i16, -2'i16])
write(stdout, s & "\n")
