# `std/complex` — construction, arithmetic and modulus on a pair of fixed complex values.
import std/syncio
import std/complex
import std/strutils

proc f(x: float): string = formatFloat(x, ffDecimal, 4)

proc go(): string =
  let a = complex(3.0, 4.0)
  let b = complex(1.0, -2.0)
  let s = a + b
  let p = a * b
  result = f(s.re) & "," & f(s.im) & "|" & f(p.re) & "," & f(p.im)
  result = result & "|" & f(abs(a))

write(stdout, go() & "\n")
