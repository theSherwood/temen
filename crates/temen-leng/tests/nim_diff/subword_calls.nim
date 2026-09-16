# Sub-word integers across a call boundary (#1488): `uint8`/`int8` params and returns, an
# accumulator wrapped repeatedly in a loop, and `div`/`mod` — which are only correct on an operand
# the producer already truncated, so this pins the canonical-width invariant, not just one operator.
import std/syncio

proc addU8(a, b: uint8): uint8 = a + b
proc addI8(a, b: int8): int8 = a + b
proc wrap16(x: uint16): uint16 = x * 3'u16

proc go(): string =
  var acc: uint8 = 0'u8
  for i in 0 ..< 10:
    acc = acc + 30'u8
  result = $addU8(200'u8, 100'u8) & "|" & $addI8(100'i8, 100'i8) & "|" & $wrap16(30000'u16)
  result = result & "|" & $acc & "|" & $(acc div 7'u8) & "|" & $(acc mod 7'u8)

write(stdout, go() & "\n")
