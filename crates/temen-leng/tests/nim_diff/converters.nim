# Converters: an implicit `converter` between two `distinct` types, applied at a call. nimony
# attaches a converter only to a type the module declares (not to `int`).
import std/syncio
type
  Meters = distinct int
  Feet = distinct int
converter toFeet(m: Meters): Feet = Feet(int(m) * 3)
proc describe(f: Feet): string = $int(f) & "ft"
let m = Meters(14)
write(stdout, describe(m) & "\n")
