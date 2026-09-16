# `std/parseopt` over an **explicit** argument list — no real argv, so both engines parse the same
# input. Covers short/long flags, `=` and `:` value forms, and positional arguments.
import std/syncio
import std/parseopt

proc go(): string =
  var p = initOptParser(@["-a", "--long=v", "--other:w", "pos1", "-bc", "pos2"])
  result = ""
  while true:
    p.next()
    case p.kind
    of cmdEnd: break
    of cmdShortOption: result = result & "S(" & p.key & "," & p.val & ")"
    of cmdLongOption: result = result & "L(" & p.key & "," & p.val & ")"
    of cmdArgument: result = result & "A(" & p.key & ")"

write(stdout, go() & "\n")
