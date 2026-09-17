# #1544: a **signed** sub-word value read from memory, then converted to its unsigned sibling.
#
# The narrow pin for the bug `std_widestrs` surfaced. A sub-word conversion shares the operand's
# machine slot, so it used to return the operand untouched on the assumption that a sub-word integer
# is always already canonical. The producers disagree: a constant `int16` is materialized
# zero-extended, the same value loaded from memory is sign-extended. Only a load with the high bit
# set tells them apart, and only an unsigned consumer notices — which is why this sat behind the
# existing sub-word cases for so long.
import std/syncio

proc go(): string =
  let neg: int16 = cast[int16](0xD83D'u16)    # high bit set
  let pos: int16 = cast[int16](0x00A9'u16)    # high bit clear — agreed even before the fix
  let p = cast[ptr UncheckedArray[int16]](alloc(16))
  p[0] = neg
  p[1] = pos
  result = $int(cast[uint16](p[0])) & "|" & $int(uint16(p[0]))
  result = result & "|" & $int(cast[uint16](p[1])) & "|" & $int(uint16(p[1]))
  # the signed reading of the same cell must keep sign-extending
  result = result & "|" & $int(int16(p[0])) & "|" & $int(int16(p[1]))
  # and the same value held in a local, which took the other producer's path
  result = result & "|" & $int(cast[uint16](neg))
  # one byte wide, the same shape
  let q = cast[ptr UncheckedArray[int8]](alloc(8))
  q[0] = cast[int8](0xA9'u8)
  result = result & "|" & $int(cast[uint8](q[0])) & "|" & $int(int8(q[0]))
  dealloc(q)
  dealloc(p)

write(stdout, go() & "\n")
