# Compile-time evaluation: a `const` computed by calling a proc, `when` on a constant, and `when`
# choosing a branch per instantiation of a generic.
import std/syncio
proc fib(n: int): int =
  if n < 2: n else: fib(n - 1) + fib(n - 2)
const F20 = fib(20)
const Big = F20 > 5000
proc kind[T](x: T): string =
  when T is int:
    result = "int"
  elif T is string:
    result = "string"
  else:
    result = "other"
var s = $F20 & "|"
when Big:
  s.add "big|"
else:
  s.add "small|"
s.add kind(1) & "," & kind("a") & "," & kind(1.5)
write(stdout, s & "\n")
