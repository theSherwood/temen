# Raw memory: `addr` of an object, a write through the pointer, a `cast` to `ptr UncheckedArray`, and
# a byte view of an `int64` (little-endian on both engines).
import std/syncio
type Pair = object
  a, b: int32
var p = Pair(a: 1, b: 2)
let q = addr p
q.b = 40
let raw = cast[ptr UncheckedArray[int32]](q)
raw[0] = 2
var x: int64 = -1
let bytes = cast[ptr UncheckedArray[uint8]](addr x)
write(stdout, $(p.a + p.b) & "|" & $bytes[0] & "|" & $sizeof(Pair) & "\n")
