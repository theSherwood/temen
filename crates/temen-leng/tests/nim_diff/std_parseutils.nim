# `std/parseutils` — the `Biggest*` parsers over trailing garbage, each returning the count consumed.
import std/syncio
import std/parseutils

proc go(): string =
  var n: BiggestInt = 0
  let used = parseBiggestInt("-1234xyz", n)
  result = $used & ":" & $n
  var h: BiggestInt = 0
  let hu = parseHex("ffz", h)
  result = result & "|" & $hu & ":" & $h
  var u: BiggestUInt = 0
  let uu = parseBiggestUInt("99 rest", u)
  result = result & "|" & $uu & ":" & $u

write(stdout, go() & "\n")
