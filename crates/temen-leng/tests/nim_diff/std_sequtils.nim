# `std/sequtils` — map/filter over a seq, the predicate folds, index queries and concat.
import std/syncio
import std/sequtils

proc dbl(x: int): int = x * 2
proc big(x: int): bool = x > 4
proc pos(x: int): bool = x > 0
proc isNine(x: int): bool = x == 9

proc go(): string =
  let xs = @[5, 3, 9, 1, 7]
  let d = xs.map(dbl)
  let f = xs.filter(big)
  result = $d.len & ":" & $d[0] & ":" & $d[4]
  result = result & "|" & $f.len & ":" & $f[0] & ":" & $f[2]
  result = result & "|" & $xs.any(isNine) & $xs.all(pos) & $xs.count(5)
  result = result & "|" & $xs.minIndex & $xs.maxIndex
  let c = concat(@[1, 2], @[3])
  result = result & "|" & $c.len & ":" & $c[2]

write(stdout, go() & "\n")
