# A module-level `let` whose initializer is an aggregate with a **non-scalar field** — the string is
# a nested SSO `oconstr`, which the const-folder used to give up on ("non-scalar-int global
# initializer", #1444). Moving the same lines inside a proc always worked, so the gap was invisible
# unless you wrote it at module level, which is a fairly normal thing to do.
import std/syncio

type Shape = object
  name: string
  area: int

let s = Shape(name: "sq", area: 16)
write(stdout, s.name & "|" & $s.area & "\n")
