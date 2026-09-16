# KNOWN GAP: #1544 — a signed sub-word value read from memory is not re-masked when converted to
# unsigned, so `ord(Utf16Char)` (`int(cast[uint16](arg))` over a `distinct int16` loaded through a
# `ptr UncheckedArray`) sees 0xFFFFD83D instead of 0xD83D. The surrogate-range checks in
# `$`(WideCString) then miss and the non-BMP round trip emits "3DA9" instead of "F09F92A9".
#
# The encode direction already agrees: both engines produce 2 UTF-16 units for U+1F4A9. Only the
# decode diverges. Promote this file back into ../ and close #1544 when it starts matching.
#
# `std/widestrs` — UTF-8 to UTF-16 and back, including a non-BMP codepoint that must surrogate-pair.
# The unit here is 16 bits wide, the sub-word territory #1488 got wrong.
import std/syncio
import std/widestrs

proc go(): string =
  var a = "hello"
  let wa = newWideCString(a)
  result = $wa.len & "|" & $wa
  var b = "h\xC3\xA9llo"                 # h, U+00E9, l, l, o
  let wb = newWideCString(b)
  result = result & "|" & $wb.len & "|" & $wb
  var c = "\xF0\x9F\x92\xA9"             # U+1F4A9, one surrogate pair in UTF-16
  let wc = newWideCString(c)
  result = result & "|" & $wc.len & "|" & $($wc == c)
  var e = ""
  result = result & "|" & $newWideCString(e).len

write(stdout, go() & "\n")
