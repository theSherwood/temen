# `std/streams` over a `StringStream` — write/read round trip, positioning and line reads. Every
# stream call is `.raises`, so the body sits in one `try`.
import std/syncio
import std/streams

proc go(): string =
  try:
    var s = newStringStream("")
    s.write("alpha\n")
    s.write("beta\n")
    s.setPosition(0)
    result = s.readLine() & "|" & s.readLine() & "|" & $s.atEnd()
    var r = newStringStream("hello world")
    result = result & "|" & r.readStr(5) & "|" & $r.getPosition()
  except:
    result = "raised"

write(stdout, go() & "\n")
