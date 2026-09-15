# Tables and sets — insertion, length, membership. Iteration order is not asserted.
import std/syncio
import std/tables
import std/sets

var t = initTable[string, int]()
t["a"] = 1
t["b"] = 2
t["a"] = 10
write(stdout, $t.len & "|" & $t.getOrDefault("a") & "|" & $t.getOrDefault("zz") & "|" & $t.hasKey("b") & "\n")

var s = initHashSet[int]()
s.incl(3)
s.incl(5)
s.incl(3)
write(stdout, $s.len & "|" & $s.contains(3) & "|" & $s.contains(4) & "\n")
