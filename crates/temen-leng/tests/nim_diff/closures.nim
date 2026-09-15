# Closures capturing mutable state, and higher-order procs. nimony needs `{.closure.}` on a nested
# proc that reads an enclosing local — without it the front-end refuses, so this is its idiom, not
# standard Nim's.
import std/syncio

proc counter(): int =
  var n = 0
  proc bump(): int {.closure.} =
    n += 2
    n
  discard bump()
  discard bump()
  bump()

proc apply(f: proc (x: int): int, v: int): int = f(v)

proc capturing(): int =
  var base = 100
  proc addBase(x: int): int {.closure.} = x + base
  addBase(5)

write(stdout, $counter() & "\n")
let triple = proc (x: int): int = x * 3
write(stdout, $apply(triple, 7) & "\n")
write(stdout, $capturing() & "\n")
