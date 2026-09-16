# `std/sets` + `std/hashes` — dedup on insert, membership, intersection, removal, and hash stability.
import std/syncio
import std/sets
import std/hashes

proc go(): string =
  var s = initHashSet[int]()
  for i in [3, 1, 4, 1, 5, 9, 2, 6, 5]:
    s.incl(i)
  var t = initHashSet[int]()
  t.incl(4)
  t.incl(9)
  t.incl(100)
  let inter = intersection(s, t)
  result = $s.len & "|" & $s.contains(9) & $s.contains(7) & "|" & $inter.len
  s.excl(9)
  result = result & "|" & $s.len & $s.contains(9)
  result = result & "|" & $(hash(42) == hash(42)) & $(hash("ab") == hash("ab"))

write(stdout, go() & "\n")
