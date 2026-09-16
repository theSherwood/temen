# KNOWN GAP — #1488: the signed half of the same hole. `int8`/`int16` results are stored back into
# their `i32` slot without a sign-extending narrow, so `127'i8 + 1` is 128 instead of -128.
import std/syncio

proc go(): string =
  let a: int8 = 127'i8
  let b: int16 = 32767'i16
  let c: int8 = 100'i8
  result = $(a + 1'i8) & "|" & $(b + 1'i16) & "|" & $(c + c) & "|" & $(a * 2'i8)

write(stdout, go() & "\n")
