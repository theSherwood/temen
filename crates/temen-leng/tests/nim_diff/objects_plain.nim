# Plain value objects: field read/write, a proc taking one by value, nested objects.
import std/syncio

type
  Point = object
    x, y: int
  Box = object
    lo, hi: Point

proc sum(p: Point): int = p.x + p.y
proc span(b: Box): int = sum(b.hi) - sum(b.lo)

proc go(): string =
  var p = Point(x: 3, y: 4)
  p.y = 10
  let b = Box(lo: Point(x: 1, y: 1), hi: Point(x: 5, y: 6))
  $p.x & "|" & $p.y & "|" & $sum(p) & "|" & $span(b)

write(stdout, go() & "\n")
