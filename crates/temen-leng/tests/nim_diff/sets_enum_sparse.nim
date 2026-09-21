# `set[enum]` where the enum's ordinals are SPARSE and reach past a 64-bit word.
#
# `sets_enum.nim` covers the easy shape: three members at ordinals 0..2, so the whole set fits in one
# machine word and membership is a single shift-and-test. nimony's OWN `TypeKind` is not that shape —
# its 57 members take explicit ordinals out of the ~700-wide shared `TagEnum`, so `set[TypeKind]` is
# an eleven-word bitset and `k in {IntT, UIntT}` has to pick a word before it picks a bit. That is
# the test `sigmatch.nim` runs on every overload candidate (#1612).
import std/syncio

type Tag = enum
  tZero = 0
  tLow = 3
  tWordEdge = 63
  tNextWord = 64
  tFar = 130
  tTop = 200

proc probe(k: Tag): string =
  if k in {tLow, tWordEdge}: "a"
  elif k in {tNextWord, tFar}: "b"
  elif k in {tTop}: "c"
  else: "z"

proc go(): string =
  result = ""
  for k in [tZero, tLow, tWordEdge, tNextWord, tFar, tTop]:
    result.add probe(k)
  var s: set[Tag] = {tNextWord, tTop}
  s.incl(tFar)
  s.excl(tNextWord)
  result.add "|" & $(tFar in s) & $(tNextWord in s) & $(tTop in s)
  result.add "|" & $s.len & "|" & $ord(tTop)

write(stdout, go() & "\n")
