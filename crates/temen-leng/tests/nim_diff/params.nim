# Parameters: `var` params mutated in place, a default argument, and a named argument.
import std/syncio
proc bump(x: var int; by = 1) = x += by
proc swap2(a, b: var string) =
  let t = a
  a = b
  b = t
var n = 10
bump(n)
bump(n, by = 5)
var a = "left"
var b = "right"
swap2(a, b)
write(stdout, $n & "|" & a & "|" & b & "\n")
