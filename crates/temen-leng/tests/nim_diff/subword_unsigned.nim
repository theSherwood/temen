# Unsigned sub-word arithmetic wraps at its declared width: `255'u8 + 1` is 0 and `0'u8 - 1` is
# 255. A `uint8`/`uint16` lives in an `i32` slot, so each `add`/`mul`/`shl` result is narrowed back
# to its width (#1488).
import std/syncio

proc go(): string =
  let b: uint8 = 255'u8
  let h: uint16 = 65535'u16
  let m: uint8 = 200'u8
  result = $(b + 1'u8) & "|" & $(h + 1'u16) & "|" & $(m + m) & "|" & $(m * 3'u8)
  result = result & "|" & $(0'u8 - 1'u8) & "|" & $(b shl 1'u8)

write(stdout, go() & "\n")
