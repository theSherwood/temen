# KNOWN GAP (#1444) — leng: a module-level `let`/`var` whose initializer is an aggregate with a
# non-scalar field fails with "non-scalar-int global initializer". Filed against `std/encodings`, but
# it bites ordinary user code: moving these three lines inside a proc is enough to make it link.
import std/syncio

type Shape = object
  name: string
  area: int

let s = Shape(name: "sq", area: 16)
write(stdout, s.name & "|" & $s.area & "\n")
