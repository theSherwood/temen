# String literals as `cstring` values: a local, an empty one, a reassigned one, an argument, a
# module-level `var` and `const`, an escape, and the literals inside a `const` object and array.
# nimony writes each literal where its value goes; the translator lays the bytes out as data and
# uses their address.
import std/syncio

type Named = object
  name: cstring
  n: int

var g: cstring = "glob"
const k: cstring = "tab\there"
const named = Named(name: "obj", n: 2)
const names: array[2, cstring] = ["x", "yz"]

proc show(c: cstring) =
  write(stdout, fromCString(c) & "|" & $c.len & "\n")

proc main =
  var s: cstring = "abc"
  var e: cstring = ""
  show(s)
  show(e)
  show("lit")
  show(k)
  s = "again"
  show(s)
  show(named.name)
  show(names[0])
  show(names[1])

main()
show(g)
