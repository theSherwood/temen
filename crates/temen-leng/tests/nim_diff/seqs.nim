# seq construction, mutation, iteration and the sequtils surface.
import std/syncio
import std/sequtils

var xs = @[5, 3, 9, 1]
xs.add(7)
var total = 0
for x in xs:
  total += x
write(stdout, $xs.len & "|" & $total & "\n")
let doubled = xs.map(proc (x: int): int = x * 2)
var dtotal = 0
for d in doubled:
  dtotal += d
write(stdout, $dtotal & "\n")
let evens = xs.filter(proc (x: int): bool = x mod 2 == 0)
write(stdout, $evens.len & "\n")
