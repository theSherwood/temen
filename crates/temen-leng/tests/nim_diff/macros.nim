# Macros: nimony builds each macro as a plugin program and runs it at compile time (natively, before
# Temen sees the code), with its own API (`newTree`, `ident`, not mainline's `newProc`/`newLit`).
# Temen runs what the macro generated, so one macro, called twice, is enough. The case is the corpus's
# slowest because `std/macros` makes the native build compile nimony's parsegen and regex plugins.
import std/[syncio, macros]
macro squareSum(a, b: untyped): untyped =
  result = newTree(nnkInfix, [ident("+"),
    newTree(nnkInfix, [ident("*"), a, a]),
    newTree(nnkInfix, [ident("*"), b, b])])
write(stdout, $squareSum(3, 4) & "|" & $squareSum(5, 12) & "\n")
