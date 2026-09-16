# `std/algorithm` — sort in place, `sorted` copy, both orders, and the `isSorted` predicate. nimony's
# algorithm has no default-`cmp` overload, so every call passes one explicitly.
import std/syncio
import std/algorithm

proc cmpInt(x, y: int): int =
  if x < y: -1
  elif x > y: 1
  else: 0

proc go(): string =
  var xs = @[5, 3, 9, 1, 7]
  xs.sort(cmpInt)
  result = ""
  for x in xs:
    result = result & $x
  result = result & "|" & $xs.isSorted(cmpInt)
  let ys = sorted(@[2, 8, 4], cmpInt)
  result = result & "|"
  for y in ys:
    result = result & $y
  let ds = sorted(@[2, 8, 4], cmpInt, SortOrder.Descending)
  result = result & "|"
  for d in ds:
    result = result & $d

write(stdout, go() & "\n")
