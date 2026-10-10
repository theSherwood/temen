# `case` over strings (with a multi-value branch), a labeled `block` left by `break` from a nested
# loop, and `continue`.
import std/syncio
proc classify(s: string): int =
  case s
  of "one": 1
  of "two", "deux": 2
  else: 0
var found = -1
block search:
  for i in 0 ..< 10:
    for j in 0 ..< 10:
      if i * j == 42:
        found = i * 10 + j
        break search
var odd = 0
var i = 0
while i < 10:
  inc i
  if i mod 2 == 0: continue
  odd += i
write(stdout, $classify("deux") & $classify("one") & $classify("x") & "|" & $found & "|" & $odd & "\n")
