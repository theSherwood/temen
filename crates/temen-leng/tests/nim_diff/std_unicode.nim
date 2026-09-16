# `std/unicode` over real multi-byte text — rune counting, case mapping and UTF-8 validation. This is
# what a signed `char` broke: runeLen counted bytes (13, not 11), toUpper skipped the accented runes,
# and validateUtf8 rejected valid input at byte 1.
import std/syncio
import std/unicode

proc go(): string =
  let s = "héllo wörld"
  result = $s.runeLen & ":" & $s.len
  result = result & "|" & s.toUpper & "|" & s.toLower
  result = result & "|" & $s.validateUtf8 & "|" & capitalize("abc")
  result = result & "|" & reversed("abc") & "|" & $runeAt(s, 0).size

write(stdout, go() & "\n")
