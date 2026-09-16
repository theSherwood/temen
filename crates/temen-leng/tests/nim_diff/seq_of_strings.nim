# `seq[string]` — a growable container whose element is itself a destructor-carrying aggregate, so
# every `add` runs a `=copy` hook and the seq's own `=destroy` runs the element's.
import std/syncio

proc go(): string =
  var xs: seq[string] = @[]
  xs.add("alpha")
  xs.add("beta")
  xs.add("gamma")
  result = ""
  for s in xs:
    result = result & s & ","
  result = result & $xs.len

write(stdout, go() & "\n")
