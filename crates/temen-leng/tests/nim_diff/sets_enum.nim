# `set[enum]` bitsets (`incl`/`excl`/`in`/`len`) and enum ordinals.
import std/syncio

type Color = enum cRed, cGreen, cBlue

proc go(): string =
  var s: set[Color] = {cRed, cBlue}
  s.incl(cGreen)
  s.excl(cRed)
  result = $(cGreen in s) & $(cRed in s) & "|" & $ord(cBlue) & "|" & $s.len

write(stdout, go() & "\n")
