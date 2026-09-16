# `std/setutils` — `fullSet`, `complement`, `toggle` and the symmetric difference over a small enum.
import std/syncio
import std/setutils

type Color = enum cRed, cGreen, cBlue

proc go(): string =
  let s = {cRed, cBlue}
  result = $(cRed in s) & $(cGreen in s) & $(cBlue in s)
  let full = fullSet(Color)
  result = result & "|" & $(cGreen in full) & "|" & $card(full)
  result = result & "|" & $card(complement(s)) & $(cGreen in complement(s))
  result = result & "|" & $card(symmetricDifference(s, {cGreen, cBlue}))
  var v = {cRed}
  v.toggle(cGreen)
  v.toggle(cRed)
  result = result & "|" & $card(v) & $(cGreen in v)

write(stdout, go() & "\n")
