# `char` is a `Narrow` cell too: `ord`/`char` round-trips, arithmetic on an ordinal, string
# building from computed chars, and char comparison. (`$c` on a char is outside nimony's subset —
# it resolves to the float32/bool overloads — so chars reach the output through `string.add`.)
import std/syncio

proc go(): string =
  let c = 'A'
  let z = char(ord(c) + 25)
  var up = ""
  for ch in "hello":
    up.add(char(ord(ch) - 32))
  var two = ""
  two.add(c)
  two.add(z)
  result = two & "|" & up & "|" & $ord('0') & "|" & $(c < 'B') & "|" & $ord(z)

write(stdout, go() & "\n")
