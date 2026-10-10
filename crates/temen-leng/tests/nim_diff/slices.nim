# Slices and backward indexes: `s[a..b]`, `s[a..^b]` and `s[^i]` on a string, and `xs[a..b]` on a
# seq. A slice is a copy, so growing it leaves its source alone.
import std/syncio
proc go(): string =
  let s = "abcdefgh"
  let xs = @[1, 2, 3, 4, 5]
  var ys = xs[1..3]
  ys.add 9
  result = s[1..3] & "|" & s[2..^2] & "|"
  result.add s[^1]
  result.add s[^3]
  result.add "|"
  for y in ys: result.add $y & ","
  result.add "|"
  for x in xs: result.add $x & ","
write(stdout, go() & "\n")
