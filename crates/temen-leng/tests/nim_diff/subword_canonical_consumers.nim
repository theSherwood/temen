# #1544, the other half: every **consumer** that depends on a sub-word value being canonically
# extended, fed from every **producer** that can hand it one.
#
# `subword_signed_to_unsigned` pins the conversion itself. This pins what the conversion exists to
# protect. `narrow_result` names the consumers that only work on a correctly-extended operand —
# `div_u`/`rem_u`/`shr_u` need zero-extension, `div_s`/`shr_s` sign-extension, comparisons both — and
# each is reached here through a pointer load, an aggregate field, a seq element and a call
# parameter, since those are different producers and nothing makes them agree by construction.
import std/syncio

type Holder = object
  a: int16
  b: int8

proc takesU16(x: uint16): int = int(x)
proc takesU8(x: uint8): int = int(x)

proc go(): string =
  let p = cast[ptr UncheckedArray[int16]](alloc(16))
  p[0] = cast[int16](0xD83D'u16)          # high bit set: the only case that tells the two apart
  let u = uint16(p[0])
  result = "shr_u=" & $int(u shr 4'u16)
  result = result & " div_u=" & $int(u div 3'u16)
  result = result & " mod_u=" & $int(u mod 7'u16)
  result = result & " gtu=" & $(u > 30000'u16)
  # the signed siblings on the same cell must keep sign-extending
  let s = p[0]
  result = result & " shr_s=" & $int(s shr 4'i16)
  result = result & " lts=" & $(s < 0'i16)
  # across a call boundary
  result = result & " param=" & $takesU16(uint16(p[0]))
  # an aggregate field, not a pointer load
  var h = Holder(a: cast[int16](0xD83D'u16), b: cast[int8](0xA9'u8))
  result = result & " fld16=" & $int(uint16(h.a)) & " fld8=" & $int(uint8(h.b))
  result = result & " fldparam=" & $takesU8(uint8(h.b))
  # a seq element
  var xs: seq[int16] = @[cast[int16](0xD83D'u16)]
  result = result & " seq=" & $int(uint16(xs[0]))
  dealloc(p)

write(stdout, go() & "\n")
