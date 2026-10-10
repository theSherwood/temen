# User-defined operators and overloading: `+`, `*` and `==` on an object, a `$` for it, and one proc
# name overloaded on its parameter types.
import std/syncio
type Vec = object
  x, y: int
proc `+`(a, b: Vec): Vec = Vec(x: a.x + b.x, y: a.y + b.y)
proc `*`(k: int, v: Vec): Vec = Vec(x: k * v.x, y: k * v.y)
proc `==`(a, b: Vec): bool = a.x == b.x and a.y == b.y
proc `$`(v: Vec): string = "(" & $v.x & "," & $v.y & ")"
proc describe(x: int): string = "int " & $x
proc describe(s: string): string = "string " & s
proc describe(v: Vec): string = "vec " & $v
let a = Vec(x: 1, y: 2)
let b = Vec(x: 10, y: 20)
let c = a + 3 * b
write(stdout, $c & "|" & $(c == Vec(x: 31, y: 62)) & "|" & $(a == b) & "|" &
  describe(7) & "|" & describe("hi") & "|" & describe(a) & "\n")
