# `openArray` params fed from both a seq and a fixed array, and a `varargs[string]` proc.
import std/syncio

proc sum(xs: openArray[int]): int =
  result = 0
  for x in xs:
    result = result + x

proc join(parts: varargs[string]): string =
  result = ""
  for p in parts:
    result = result & p

proc go(): string =
  let s = @[1, 2, 3]
  let a = [10, 20]
  $sum(s) & "|" & $sum(a) & "|" & join("x", "y", "z")

write(stdout, go() & "\n")
