# `char` is **unsigned** 0..255. leng classified nim's `(c 8)` as signed, so `ord(s[i])` on a UTF-8
# continuation byte read -61 instead of 195 and every `>= 0x80` test went false. Note `ord(c) and 0xC0`
# is right either way (`-61 and 0xC0` == `0xC0`): only the *comparisons* were wrong, which is how this
# survived behind a green `std/unicode` import. See std_unicode.nim for the user-visible damage.
import std/syncio

proc go(): string =
  let s = "hé"          # 'h' = 0x68, 'é' = 0xC3 0xA9
  result = $s.len
  for i in 0 ..< s.len:
    result = result & ":" & $ord(s[i])
  let c = s[1]
  result = result & "|" & $(ord(c) and 0xC0) & "|" & $(ord(c) >= 0x80)
  result = result & "|" & $(c > '\127')

write(stdout, go() & "\n")
