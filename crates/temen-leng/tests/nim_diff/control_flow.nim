# Branching, loops, early exit, and nested/recursive procs.
import std/syncio

proc classify(n: int): string =
  if n < 0: "neg"
  elif n == 0: "zero"
  else: "pos"

proc fib(n: int): int =
  if n < 2: n else: fib(n - 1) + fib(n - 2)

write(stdout, classify(-3) & "|" & classify(0) & "|" & classify(7) & "\n")
write(stdout, $fib(15) & "\n")

var acc = 0
for i in 1 .. 10:
  if i mod 3 == 0: continue
  if i > 8: break
  acc += i
write(stdout, $acc & "\n")

var k = 0
var guard = 0
while k < 100:
  k += 7
  guard += 1
write(stdout, $k & "|" & $guard & "\n")

let day = 3
let name = case day
  of 1: "mon"
  of 2: "tue"
  of 3: "wed"
  else: "other"
write(stdout, name & "\n")
