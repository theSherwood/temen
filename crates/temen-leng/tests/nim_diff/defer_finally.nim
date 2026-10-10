# `defer` and `try`/`finally` on both the normal and the raising path, under nimony's ErrorCode
# exceptions (see exceptions.nim). The log records the order they ran in.
import std/syncio
var log = ""
proc work(fail: bool): int {.raises.} =
  defer: log.add "d"
  try:
    if fail:
      raise ValueError
    result = 1
  finally:
    log.add "f"
proc go() =
  try:
    discard work(false)
    discard work(true)
  except ErrorCode:
    log.add "e"
go()
write(stdout, log & "\n")
