# `std/bitops` over a `uint32` — population/scan counts, rotation, the bit tests and the bitwise set.
import std/syncio
import std/bitops

proc go(): string =
  let x = 0b1011_0010'u32
  result = $countSetBits(x) & "|" & $firstSetBit(x) & "|" & $trailingZeroBits(x)
  result = result & "|" & $leadingZeroBits(x) & "|" & $rotateLeftBits(x, 4)
  result = result & "|" & $bitand(x, 0xF0'u32) & "|" & $bitor(x, 1'u32) & "|" & $bitxor(x, x)
  result = result & "|" & $testBit(x, 1) & $testBit(x, 0) & "|" & $parityBits(x)

write(stdout, go() & "\n")
