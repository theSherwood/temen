# `std/tables` ordered: `OrderedTable` keeps insertion order through an overwrite and a delete,
# so its iteration can be asserted (unlike `Table`, see tables_sets.nim).
import std/[syncio, tables]
proc go(): string {.raises.} =
  var t = initOrderedTable[string, int]()
  t["zeta"] = 1
  t["alpha"] = 2
  t["mid"] = 3
  t["alpha"] = 20
  result = ""
  for k, v in t: result.add k & "=" & $v & ","
  t.del "zeta"
  result.add "|" & $t.len & "|" & $t["mid"]
proc main() =
  try:
    write(stdout, go() & "\n")
  except ErrorCode:
    write(stdout, "err\n")
main()
