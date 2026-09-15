# `ref object`: allocation, field access through the reference, and mutation through an alias.
import std/syncio

type Shape = ref object
  name: string
  area: int

proc grow(s: Shape) = s.area = s.area * 2

proc go(): string =
  let s = Shape(name: "sq", area: 16)
  grow(s)
  let alias = s
  alias.area = alias.area + 1
  s.name & "|" & $s.area

write(stdout, go() & "\n")
