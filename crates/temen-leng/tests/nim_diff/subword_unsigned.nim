# KNOWN GAP — #1488: sub-word integer arithmetic is never truncated to its declared width. A `uint8`
# local lives in an `i32` SSA slot and the result of an `add`/`mul`/`shl` is stored back un-narrowed,
# so `255'u8 + 1` is 256 instead of 0 and `0'u8 - 1` leaks the whole 32-bit slot (4294967295).
import std/syncio

proc go(): string =
  let b: uint8 = 255'u8
  let h: uint16 = 65535'u16
  let m: uint8 = 200'u8
  result = $(b + 1'u8) & "|" & $(h + 1'u16) & "|" & $(m + m) & "|" & $(m * 3'u8)
  result = result & "|" & $(0'u8 - 1'u8) & "|" & $(b shl 1'u8)

write(stdout, go() & "\n")
