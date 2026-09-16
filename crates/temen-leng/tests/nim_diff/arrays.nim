# Fixed-size arrays: of ints, of strings, as an object field, mutated by index and walked by `for`.
import std/syncio

type Grid = object
  cells: array[4, int]
  tag: string

proc go(): string =
  var a: array[3, int] = [10, 20, 30]
  a[1] = 25
  var names: array[2, string] = ["x", "y"]
  var g = Grid(cells: [1, 2, 3, 4], tag: "grid")
  result = $a[0] & $a[1] & $a[2] & "|" & names[0] & names[1] & "|"
  var s = 0
  for c in g.cells:
    s = s + c
  result = result & g.tag & $s

write(stdout, go() & "\n")
