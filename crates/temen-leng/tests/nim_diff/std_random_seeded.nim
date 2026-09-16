# `std/random` from a **fixed seed** — the generator is deterministic, so the whole sequence is a
# fixture. This is 64-bit PRNG arithmetic (shifts, rotations, wrapping multiplies) end to end.
import std/syncio
import std/random

proc go(): string =
  var r = initRand(42'i64)
  result = $r.next()
  result = result & "|" & $r.next()
  result = result & "|" & $r.rand(100)
  result = result & "|" & $r.rand(100)
  var r2 = initRand(42'i64)
  result = result & "|" & $(r2.next() == 0'u64)
  var r3 = initRand(7'i64)
  result = result & "|" & $r3.rand(1000) & "," & $r3.rand(1000) & "," & $r3.rand(1000)

write(stdout, go() & "\n")
