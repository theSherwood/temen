# `std/options` (some/none/get) and `std/intsets` (incl/excl/containsOrIncl).
import std/syncio
import std/options
import std/intsets

proc go(): string =
  let a = some(42)
  let b = none[int]()
  result = $a.isSome & $b.isNone & "|" & $a.get & "|" & $a.unsafeGet

  var s = initIntSet()
  for i in [7, 3, 7, 11]:
    s.incl(i)
  result = result & "|" & $s.contains(7) & $s.contains(8)
  s.excl(7)
  result = result & "|" & $s.contains(7)
  result = result & "|" & $s.containsOrIncl(3) & $s.containsOrIncl(99)

write(stdout, go() & "\n")
