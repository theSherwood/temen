# Inline iterators: a counting `iterator` driven by `for`, and one yielding a tuple destructured
# at the loop.
import std/syncio
iterator countTo(n: int): int =
  var i = 1
  while i <= n:
    yield i
    inc i
iterator pairsOf(xs: seq[int]): (int, int) =
  var i = 0
  while i < xs.len:
    yield (i, xs[i])
    inc i
var acc = 0
for x in countTo(5): acc += x
var s = $acc
for i, v in pairsOf(@[10, 20, 30]): s.add "|" & $i & ":" & $v
write(stdout, s & "\n")
