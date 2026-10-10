# Templates: an `untyped` statement-block template that repeats its body, and an expression
# template.
import std/syncio
template twice(body: untyped) =
  body
  body
template sq(x: int): int = x * x
var n = 0
twice:
  n += sq(3)
write(stdout, $n & "\n")
