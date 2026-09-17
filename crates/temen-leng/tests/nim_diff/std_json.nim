# `std/json` — parse a literal document, look keys up through `{}`, walk an array, and re-serialize.
# nimony's json is a cursor API over a `JsonTree`, not mainline Nim's ref-node tree.
import std/syncio
import std/json

proc go(): string =
  var t = parseJson("""{"name":"temen","n":42,"ok":true,"xs":[1,2,3]}""")
  result = t{"name"}.getStr & "|" & $t{"n"}.getInt & "|" & $t{"ok"}.getBool
  result = result & "|" & $t{"n"}.kind & "|" & $t{"name"}.kind
  var total: int64 = 0
  var count = 0
  for e in items(t{"xs"}):
    total = total + e.getInt
    count = count + 1
  result = result & "|" & $count & ":" & $total
  result = result & "|" & $t.hasError

write(stdout, go() & "\n")
