# Signed sub-word arithmetic wraps at its declared width: `127'i8 + 1` is -128. An `int8`/`int16`
# lives in an `i32` slot, so each result is sign-extended back to its width (#1488).
import std/syncio

proc go(): string =
  let a: int8 = 127'i8
  let b: int16 = 32767'i16
  let c: int8 = 100'i8
  result = $(a + 1'i8) & "|" & $(b + 1'i16) & "|" & $(c + c) & "|" & $(a * 2'i8)

write(stdout, go() & "\n")
