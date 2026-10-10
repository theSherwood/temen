# Closure iterators: `{.closure.}` keeps the iterator's state between yields, as a first-class
# value bound to a `let` and called with its arguments.
import std/syncio
iterator gen(n: int): int {.closure.} =
  var i = 0
  while i < n:
    yield i * i
    inc i
proc go(): string =
  let it = gen
  result = ""
  for x in it(4): result.add $x & ","
write(stdout, go() & "\n")
