# String building, comparison and the strutils surface that does not raise.
import std/syncio
import std/strutils

let s = "Hello, Temen"
write(stdout, s & "|" & $s.len & "|" & toUpperAscii(s) & "|" & toLowerAscii(s) & "\n")
write(stdout, $s.contains("Temen") & "|" & $s.startsWith("Hello") & "|" & $s.endsWith("en") & "\n")
write(stdout, strip("  pad  ") & "|" & repeat("ab", 3) & "|" & replace(s, "Temen", "World") & "\n")
var acc = ""
for i in 0 ..< 5:
  acc.add($i)
write(stdout, acc & "|" & $(acc == "01234") & "|" & $("abc" < "abd") & "\n")
