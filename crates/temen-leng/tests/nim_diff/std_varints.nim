# `std/varints` — LEB128-style round trip across the byte-length boundaries.
import std/syncio
import std/varints

proc roundtrip(x: uint64): string =
  var buf: array[16, byte] = default(array[16, byte])
  let n = writeVu64(buf, x)
  var back: uint64 = 0
  let m = readVu64(buf, back)
  result = $n & ":" & $m & ":" & $back

proc go(): string =
  result = roundtrip(0'u64)
  result = result & "|" & roundtrip(127'u64)
  result = result & "|" & roundtrip(128'u64)
  result = result & "|" & roundtrip(300'u64)
  result = result & "|" & roundtrip(0xFFFF_FFFF'u64)

write(stdout, go() & "\n")
