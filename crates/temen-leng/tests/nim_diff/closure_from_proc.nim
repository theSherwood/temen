# A plain proc where a `{.closure.}` is expected (#2611). nimony converts the proc by casting it to
# the closure's function type, beside a nil env, and a call through a closure-typed field passes the
# env to it all the same: this unit's procs, a sibling module's (`math.fac`), one returning an object,
# one returning nothing, a real closure beside them, and closure equality. `std/times` does this for
# its UTC zone.
import std/[syncio, math]

type
  Pair = object
    a, b: int
  Z = object
    f: proc (x: int): int {.closure.}

proc plain(x: int): int = x + 1
proc pair(x: int): Pair = Pair(a: x, b: x * 2)
proc bump(total: var int) = total += 7

proc mk(f: proc (x: int): int {.closure.}): Z = Z(f: f)
proc call(z: Z; x: int): int = z.f(x)
proc callPair(f: proc (x: int): Pair {.closure.}; x: int): Pair = f(x)
proc callVoid(f: proc (total: var int) {.closure.}; total: var int) = f(total)

let z = mk(plain)

proc go(): string =
  var k = 3
  let lib = mk(fac)
  let real = mk(proc (x: int): int {.closure.} = x * k)
  let p = callPair(pair, 5)
  var total = 1
  callVoid(bump, total)
  callVoid(bump, total)
  # A proc converted twice is one closure: the conversion keeps the proc's own identity.
  let again = mk(plain)
  result = $call(z, 41) & "|" & $call(lib, 5) & "|" & $call(real, 7) & "|" & $p.a & "," & $p.b &
    "|" & $total & "|" & $(again.f == z.f) & $(lib.f == z.f)

write(stdout, go() & "\n")
