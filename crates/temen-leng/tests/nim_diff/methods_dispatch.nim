# Methods: dynamic dispatch over a `ref object of RootObj` hierarchy, called through a `seq` of the
# base type, with a `{.base.}` method and two overrides.
import std/syncio
type
  Shape = ref object of RootObj
    name: string
  Square = ref object of Shape
    side: int
  Circle = ref object of Shape
    r: int
method area(s: Shape): int {.base.} = 0
method area(s: Square): int = s.side * s.side
method area(s: Circle): int = 3 * s.r * s.r
proc go(): string =
  let shapes: seq[Shape] = @[Shape(Square(name: "sq", side: 3)), Shape(Circle(name: "c", r: 2))]
  result = ""
  for s in shapes: result.add s.name & "=" & $area(s) & ";"
write(stdout, go() & "\n")
