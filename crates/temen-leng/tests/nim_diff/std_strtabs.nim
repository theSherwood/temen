# `std/strtabs` — a string-keyed table in both case modes. Every cell is a heap string inside an
# aggregate, the shape that #1480 got wrong.
import std/syncio
import std/strtabs

proc go(): string =
  var t = newStringTable(modeCaseSensitive)
  t["alpha"] = "1"
  t["Beta"] = "2"
  t["alpha"] = "overwritten"
  result = t.getOrDefault("alpha") & "|" & t.getOrDefault("Beta")
  result = result & "|" & $t.len & "|" & $t.hasKey("beta") & $t.hasKey("Beta")
  var ci = newStringTable(modeCaseInsensitive)
  ci["Gamma"] = "3"
  result = result & "|" & ci.getOrDefault("gamma") & ci.getOrDefault("GAMMA")
  result = result & "|" & ci.getOrDefault("absent", "fallback")

write(stdout, go() & "\n")
