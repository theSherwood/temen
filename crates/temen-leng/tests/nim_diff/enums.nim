# Enums: `$` gives the name, `ord`/`succ`/`pred`/`low`/`high`, a range over the type, an enum with
# explicit values and one with string names, and `case` over an enum.
import std/syncio
type
  Color = enum red, green, blue
  Code = enum ok = 0, notFound = 404, teapot = 418
  Dir = enum north = "N", south = "S"
proc warm(c: Color): bool =
  case c
  of red: true
  of green, blue: false
var s = ""
for c in low(Color)..high(Color): s.add $c & ":" & $ord(c) & ","
s.add "|" & $succ(red) & "," & $pred(blue) & "," & $low(Color) & "," & $high(Color)
s.add "|" & $notFound & "=" & $ord(notFound) & "," & $teapot & "," & $ok
s.add "|" & $north & $south & "|" & $warm(red) & $warm(blue)
write(stdout, s & "\n")
