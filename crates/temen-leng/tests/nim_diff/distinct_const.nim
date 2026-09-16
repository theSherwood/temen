# `distinct` types and module-level `const` of three shapes — scalar, string, and array.
import std/syncio

type Meters = distinct int

const
  Limit = 42
  Greeting = "hi"
  Table3 = [1, 2, 3]

proc go(): string =
  let m = Meters(7)
  $int(m) & "|" & $Limit & "|" & Greeting & "|" & $Table3[2]

write(stdout, go() & "\n")
